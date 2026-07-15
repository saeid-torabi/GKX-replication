# Thesis Replication

This repository is a working replication pipeline for Gu, Kelly, and Xiu, *Empirical Asset Pricing via Machine Learning*.

The current codebase is organized around two stages:

1. Data preparation in the notebook.
2. Recursive model training and evaluation in Python scripts.

## Project Goal

The aim is to replicate the stock-level forecasting setup from the paper:

- 94 firm characteristics
- 8 macro predictors
- 74 SIC2 industry dummies
- 920 baseline predictors after adding characteristic-macro interactions

The training pipeline follows the paper's recursive logic:

- expanding training window
- rolling validation window
- one-year out-of-sample test window

The current main pipeline focuses on recursive stock-level prediction and stock-level out-of-sample evaluation.

## What The Code Does So Far

The current codebase already does the following:

1. Takes a cleaned base parquet panel produced in the notebook.
2. Streams that parquet in batches instead of loading the full interaction matrix into memory.
3. Rebuilds the 920 GKX baseline predictors on the fly:
   - 94 firm characteristics
   - 752 characteristic-macro interactions
   - 74 SIC2 dummies
4. Runs recursive yearly experiments:
   - expanding training window
   - rolling 12-year validation window
   - one-year out-of-sample test window
5. Trains a neural network for each test year using an MSE objective plus
   GKX-style L1 regularization on network weight tensors.
6. Optionally trains a GKX-style ensemble of independently initialized neural
   networks and averages their out-of-sample forecasts.
7. Selects the best model state for each ensemble member using validation loss.
8. Produces out-of-sample stock predictions for the test window.
9. Computes stock-level out-of-sample \(R^2\).
10. Saves predictions, split summaries, evaluation tables, and plots.

So, in practical terms, the code now runs a full recursive forecasting experiment and writes the outputs to an experiment folder.

## Repository Structure

- [data_wrangling.ipynb](./data_wrangling.ipynb)
  Builds the cleaned base panel and saves it to parquet.

- [data_generator.py](./data_generator.py)
  Streams the base parquet in batches and constructs the 920 predictors on the fly.

- [models.py](./models.py)
  Defines the feed-forward neural network architectures.

- [train.py](./train.py)
  Handles model fitting, validation loss evaluation, best-state selection, and out-of-sample prediction.

- [splits.py](./splits.py)
  Generates recursive train/validation/test windows.

- [evaluate.py](./evaluate.py)
  Computes out-of-sample predictive metrics such as stock-level \(R^2_{oos}\).

- [backtest.py](./backtest.py)
  Experimental decile-backtest utilities. These are currently not part of the main reporting pipeline because they are not aligned with the paper's Table 5 methodology.

- [main.py](./main.py)
  Main recursive experiment entrypoint. It orchestrates split generation, training, prediction, evaluation, backtesting, and plot saving.

## Data Workflow

The notebook is expected to produce a lean base parquet file, currently:

- `gkx_base_dataset_2016.parquet`

That file should contain:

- `permno`
- `YYYYMM`
- `excess_ret`
- 94 cleaned firm characteristics
- 74 `sic2_` dummy columns
- 8 macro columns prefixed with `macro_`

The interaction matrix is **not** saved explicitly in the parquet file. It is generated batch-by-batch during training to avoid memory failures.

## Current Python Workflow

When [main.py](./main.py) runs, it does this:

1. Parses the experiment arguments.
2. Builds the recursive yearly train/validation/test splits from [splits.py](./splits.py).
3. For each test year:
   - creates a filtered [GKXDataGenerator](./data_generator.py) for train, validation, and test
   - builds the requested neural network from [models.py](./models.py)
   - trains the configured number of ensemble members with [train.py](./train.py),
     including the configured L1 penalty coefficient
   - selects the best model state for each member by validation loss
   - predicts on the out-of-sample test year with each member
   - averages ensemble member predictions before evaluation
4. Concatenates all averaged out-of-sample predictions across years.
5. Computes stock-level out-of-sample \(R^2\) with [evaluate.py](./evaluate.py).
6. Saves:
   - predictions parquet
   - split-level summary CSV
   - monthly and annual \(R^2\) tables
   - decile portfolio performance table
   - JSON summary
   - combined learning-curves plot, with the penalized training objective when
     L1 regularization is active

## Current Preprocessing Logic

The data preparation pipeline is intended to follow the paper's logic:

1. Align dates across datasets.
2. Compute monthly stock excess returns.
3. Merge returns and characteristics.
4. Restrict the sample to the paper window ending in 2016.
5. Cross-sectionally rank characteristics month by month into `[-1, 1]`.
6. Impute remaining characteristic missings with monthly cross-sectional medians.
7. Fill unresolved characteristic missings with `0`.
8. Create SIC2 dummies.
9. Merge the 8 macro predictors.
10. Save the cleaned base panel to parquet.

## Installation

The current Python pipeline requires:

- `numpy`
- `pandas`
- `pyarrow`
- `torch`
- `matplotlib`

Install the missing runtime dependencies with:

```bash
python -m pip install numpy pandas pyarrow torch matplotlib
```

## Quick Smoke Test

Run a fast one-year check (single network, no tuning) to confirm the pipeline
works end to end:

```bash
python main.py --model NN1 --epochs 5 --ensemble_size 1 --max_test_years 1 --output_dir /tmp/gkx_smoke
```

This checks that:

- recursive split generation works
- parquet streaming works
- interaction construction works
- the model trains
- predictions are generated
- evaluation files are written

## Example Full Run

Example experiment run:

```bash
python main.py --model NN1 --epochs 20 --batch_size 8192 --l1_lambda 1e-5 --ensemble_size 5 --seed 42 --output_dir outputs/nn1_ensemble_full
```

The `--l1_lambda` option controls the L1 coefficient in the neural-network
training objective. It defaults to `1e-5`, the lower end of the GKX tuning
range. Set `--l1_lambda 0` to run an unregularized diagnostic, or rerun with
larger values such as `1e-4` and `1e-3` when tuning by validation performance.

The `--ensemble_size` option controls how many independently initialized neural
networks are trained for each recursive split. Member `k` uses seed
`--seed + k`, with `k` starting at zero. The final stock-level forecast is the
simple average of member forecasts, following the GKX neural-network ensemble
approach for reducing prediction variance from stochastic optimization.

## Validation Grid Search

The main entrypoint can tune neural-network hyperparameters on each split's
rolling validation window before scoring that split's test year:

```bash
python main.py \
  --model NN1 \
  --epochs 100 \
  --batch_size 10000 \
  --tune_hyperparameters \
  --tune_learning_rates 0.001,0.01 \
  --tune_l1_lambdas 1e-5,3e-5,1e-4,3e-4,1e-3 \
  --ensemble_size 5 \
  --seed 42 \
  --test_start_year 1987 \
  --test_end_year 2016 \
  --early_stopping_patience 5 \
  --decile_weight_col market_cap \
  --output_dir outputs/nn1_tuned_ensemble5_full
```

For each recursive split, every learning-rate/L1 combination is trained on the
expanding training window and evaluated on that split's 12-year validation
window. The combination with the lowest MSE from the averaged ensemble
validation forecast is selected, and only that selected ensemble is used for
the test-year prediction. Individual member validation losses are still saved
as diagnostics. The selected hyperparameters are saved in the split results
CSV, and the full validation grid is saved to `<model>_tuning_results.csv`.

## Checkpointing and Resuming

Long tuned-ensemble runs (many networks per test year) are now crash-safe. A run
writes checkpoints as it goes and **auto-resumes** if it is restarted, so an app
close, a laptop restart, or a failure at hour 200 no longer throws away the work.

How it works:

- A `checkpoints/` folder is created inside `--output_dir`.
- Training is checkpointed at **ensemble-member granularity**. After each member
  of each hyperparameter combination is trained, its weights and history are
  saved. After a combination finishes, a `combo.json` marker records its
  validation loss. After a full test year finishes, that year's predictions and
  metric rows are persisted and `progress.json` is updated.
- On restart with the **same command**, the run reads `progress.json`, skips
  every completed test year, reloads any already-trained members/combos for the
  in-progress year, and trains only what is missing.
- **Output files are rewritten after every completed test year**, so
  `nn1_predictions.parquet`, the R^2 tables, portfolio performance, summary JSON,
  and plots always reflect all years finished so far — even if the run is
  interrupted before reaching the final year.

Config guard: the run's key settings (model, data, epochs, learning rate, L1,
tuning grid, ensemble size, seed, window, etc.) are stored in
`checkpoints/run_config.json`. If you restart with **different** settings, the
run refuses to resume and tells you what changed, so completed and new years are
never silently mixed. Use `--force_restart` to discard existing checkpoints and
begin again from the first test year.

Because checkpoints are keyed to `--output_dir`, just re-run the exact same
command after any interruption:

```bash
python main.py --model NN1 --epochs 100 --batch_size 10000 \
  --tune_hyperparameters --tune_learning_rates 0.001,0.01 \
  --tune_l1_lambdas 1e-5,3e-5,1e-4,3e-4,1e-3 \
  --ensemble_size 5 --seed 42 \
  --test_start_year 1987 --test_end_year 2016 \
  --output_dir outputs/nn1_tuned_ensemble5_full
# ... if it dies, run the identical command again to continue.
```

### Testing the checkpoint logic

- `python test_checkpoint_logic.py` — fast checks of the resume state machine
  (config guard, progress tracking, idempotent output rewrites, and
  combo/member skip-on-resume). Runs without torch or the real dataset.
- `python smoke_test_checkpoints.py` — end-to-end test on tiny synthetic data:
  it does a clean run, injects a crash mid-year, resumes, and verifies the
  resume trains only the missing networks and reproduces the clean run's
  predictions exactly. Requires torch + pyarrow (i.e. run it on your machine).

## Performance and device selection

Two options control training speed, both chosen to be paper-faithful by default:

- `--device {auto,cpu,mps,cuda}` (default `auto`, which picks mps > cuda > cpu).
  Because the neural nets here are small, the per-batch overhead of Apple's MPS
  backend can exceed its benefit, and on an 8 GB Mac MPS also competes for
  unified memory. On Apple silicon, `--device cpu` (optionally with a larger
  `--batch_size`, e.g. 30000) is often materially faster. On an NVIDIA machine
  use `--device cuda`. The device is not part of the resume config guard, so a
  run checkpointed on one machine can be continued on another.

- Tune-then-ensemble (default). During validation grid search, each
  hyperparameter candidate is evaluated by training a **single** network; the
  full `--ensemble_size` ensemble is then trained **only at the selected
  configuration** (member 0 from the grid is reused). This trains roughly
  `n_grid + ensemble_size` networks per year instead of
  `n_grid x ensemble_size`, a large saving with negligible effect on the final
  forecast. The paper does not specify the interaction between tuning and
  ensembling, so this is an efficiency choice, not a claim about the paper.
  Pass `--full_ensemble_grid` to instead train the full ensemble at every grid
  point (the older, more expensive behavior).

Example, tuned CPU run on Apple silicon:

```bash
python main.py --model NN1 --device cpu --batch_size 30000 \
  --tune_hyperparameters --tune_learning_rates 0.001,0.01 \
  --tune_l1_lambdas 1e-5,3e-5,1e-4,3e-4,1e-3 \
  --ensemble_size 10 --seed 42 \
  --test_start_year 1987 --test_end_year 2016 \
  --output_dir outputs/nn1_tuned_ensemble10_full
```

The console shows a compact, hierarchical progress view: a header with the run
configuration, then per test year a one-line window summary, one line per tuning
candidate (`[k/N] lr=… l1=… val=…`), the selected configuration, one line per
ensemble network, and a one-line result. While a network trains, a heartbeat dot
is printed per completed epoch so a long, otherwise silent run visibly shows it
is alive.

To measure where time goes on your own hardware before a long run:

```bash
python profile_pipeline.py --device cpu --batch_size 30000
```

## Outputs

Each experiment writes output files such as:

- predictions parquet
- split-level validation and test summary
- monthly out-of-sample \(R^2\)
- annual out-of-sample \(R^2\)
- selected-split learning history with train MSE, train objective, validation
  MSE, best epoch, and patience counter
- decile portfolio performance table with prediction, average return, return
  standard deviation, and annualized Sharpe ratio for deciles 1-10 plus 10-1
- JSON summary
- learning curves PNG

## Notes

- The paper's Table 5 is a portfolio-level predictive-\(R^2\) exercise, not the same thing as the decile long-short backtest utility.
- The main pipeline reports decile long-short diagnostics, but they are not a
  substitute for the paper's Table 5 portfolio-level predictive-\(R^2\) exercise.
- The neural-network training loop now supports GKX-style L1 regularization.
  Training logs report both plain training MSE and the penalized training
  objective; validation selection remains based on unpenalized validation MSE.
  Learning-curve plots show the full penalized training objective when it is
  available.
- Neural-network runs support GKX-style seed ensembles through
  `--ensemble_size`. The saved predictions are the averaged ensemble forecasts,
  and split-level validation loss reports the mean, minimum, and maximum best
  validation loss across ensemble members.
- Split-level validation grid search can be enabled with
  `--tune_hyperparameters`; the supported grid currently covers Adam learning
  rate and L1 penalty, which are the neural-network tuning parameters listed in
  the paper's Internet Appendix table. When ensembles are used, grid selection
  is based on the validation MSE of the averaged ensemble forecast.
- The current neural network code is an initial implementation and should still be checked carefully against the paper's full hyperparameter and ensemble choices before treating results as final.
- The current training loop keeps the best epoch by validation loss and supports early stopping with patience.
- Runs save `<model>_learning_history.csv` so early stopping can be audited
  numerically instead of only from the learning-curve plot.
- The current setup is closer to a strong research prototype than a final paper-faithful production pipeline.

## Reference

Gu, Shihao, Bryan Kelly, and Dacheng Xiu. *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 2020.

Paper:

- https://academic.oup.com/rfs/article/33/5/2223/5758276

Internet appendix:

- https://academic.oup.com/rfs/article/33/5/2223/5758276#supplementary-data




## Full Test Command

```bash

python main.py \
  --model NN1 \
  --device cpu \
  --batch_size 10000 \
  --epochs 100 \
  --early_stopping_patience 5 \
  --tune_hyperparameters \
  --tune_learning_rates 0.001,0.01 \
  --tune_l1_lambdas 1e-5,3e-5,1e-4,3e-4,1e-3 \
  --ensemble_size 10 \
  --seed 42 \
  --validation_years 12 \
  --test_start_year 1987 \
  --test_end_year 2016 \
  --decile_weight_col market_cap \
  --output_dir outputs/nn1_tuned_ensemble10_full_1987_2016

# ... if it dies, run the identical command again to continue.
```
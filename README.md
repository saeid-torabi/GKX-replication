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
5. Trains a neural network for each test year.
6. Selects the best model state using validation loss.
7. Produces out-of-sample stock predictions for the test window.
8. Computes stock-level out-of-sample \(R^2\).
9. Saves predictions, split summaries, evaluation tables, and plots.

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
   - trains the model with [train.py](./train.py)
   - selects the best model state by validation loss
   - predicts on the out-of-sample test year
4. Concatenates all out-of-sample predictions across years.
5. Computes stock-level out-of-sample \(R^2\) with [evaluate.py](./evaluate.py).
6. Saves:
   - predictions parquet
   - split-level summary CSV
   - monthly and annual \(R^2\) tables
   - JSON summary
   - combined learning-curves plot

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

Run a one-year, one-batch smoke test:

```bash
python main.py --model NN1 --epochs 1 --batch_size 2048 --max_train_batches 1 --max_val_batches 1 --max_test_years 1 --output_dir /tmp/gkx_smoke
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
python main.py --model NN1 --epochs 20 --batch_size 8192 --output_dir outputs/nn1_full
```

## Outputs

Each experiment writes output files such as:

- predictions parquet
- split-level validation and test summary
- monthly out-of-sample \(R^2\)
- annual out-of-sample \(R^2\)
- JSON summary
- learning curves PNG

## Notes

- The paper's Table 5 is a portfolio-level predictive-\(R^2\) exercise, not the same thing as the decile long-short backtest utility.
- To avoid reporting a non-comparable Sharpe ratio, the main pipeline currently does not report decile-backtest results.
- The current neural network code is an initial implementation and should still be checked carefully against the paper's exact hyperparameter and tuning choices before treating results as final.
- The current training loop keeps the best epoch by validation loss, but it does **not** yet implement true early stopping with patience.
- The current setup is closer to a strong research prototype than a final paper-faithful production pipeline.

## Reference

Gu, Shihao, Bryan Kelly, and Dacheng Xiu. *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 2020.

Paper:

- https://academic.oup.com/rfs/article/33/5/2223/5758276

Internet appendix:

- https://academic.oup.com/rfs/article/33/5/2223/5758276#supplementary-data

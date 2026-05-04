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

The portfolio backtest currently forms monthly long-short decile portfolios from out-of-sample stock predictions.

## Repository Structure

- [data_wrangling.ipynb](./data_wrangling.ipynb)
  Builds the cleaned base panel and saves it to parquet.

- [data_generator.py](./data_generator.py)
  Streams the base parquet in batches and constructs the 920 predictors on the fly.

- [models.py](./models.py)
  Defines the feed-forward neural network architectures.

- [train.py](./train.py)
  Handles model fitting, validation loss evaluation, and out-of-sample prediction.

- [splits.py](./splits.py)
  Generates recursive train/validation/test windows.

- [evaluate.py](./evaluate.py)
  Computes out-of-sample predictive metrics such as stock-level \(R^2_{oos}\).

- [backtest.py](./backtest.py)
  Builds decile portfolios from out-of-sample predictions and computes long-short performance.

- [main.py](./main.py)
  Main experiment entrypoint.

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

Install the missing runtime dependencies with:

```bash
python -m pip install numpy pandas pyarrow torch
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
- evaluation and backtest files are written

## Example Full Run

Example experiment run:

```bash
python main.py --model NN1 --epochs 5 --batch_size 8192 --output_dir outputs/nn1_full
```

## Outputs

Each experiment writes output files such as:

- predictions parquet
- split-level validation and test summary
- monthly out-of-sample \(R^2\)
- annual out-of-sample \(R^2\)
- decile portfolio returns
- long-short 10 minus 1 returns
- JSON summary

## Notes

- The current backtest defaults to equal-weighted deciles unless a valid positive weight column is available and passed through the pipeline.
- The paper's stock-level decile portfolios are reconstituted monthly.
- The current neural network code is an initial implementation and should still be checked carefully against the paper's exact hyperparameter and tuning choices before treating results as final.

## Reference

Gu, Shihao, Bryan Kelly, and Dacheng Xiu. *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 2020.

Paper:

- https://academic.oup.com/rfs/article/33/5/2223/5758276

Internet appendix:

- https://academic.oup.com/rfs/article/33/5/2223/5758276#supplementary-data

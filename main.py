import argparse
import json
import time
from pathlib import Path

import pandas as pd

from backtest import annualized_sharpe_ratio, build_long_short_deciles
from data_generator import GKXDataGenerator
from evaluate import summarize_prediction_panel
from models import build_neural_net
from splits import generate_gkx_splits
from train import predict_model, train_model


def _metadata_cols(weight_col):
    base_cols = ["permno", "YYYYMM"]
    if weight_col:
        base_cols.append(weight_col)
    return base_cols


def main():
    parser = argparse.ArgumentParser(description="GKX recursive training entrypoint")
    parser.add_argument(
        "--model",
        type=str,
        default="NN1",
        help="Network preset to train: NN1, NN2, NN3, NN4, or NN5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per recursive split.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of rows to stream from parquet per batch.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="gkx_base_dataset_2016.parquet",
        help="Path to the lean GKX base-panel parquet file.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--test_start_year",
        type=int,
        default=1987,
        help="First out-of-sample test year.",
    )
    parser.add_argument(
        "--test_end_year",
        type=int,
        default=2016,
        help="Last out-of-sample test year.",
    )
    parser.add_argument(
        "--validation_years",
        type=int,
        default=12,
        help="Width of the rolling validation window in years.",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional cap on training batches per epoch for smoke tests.",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per epoch for smoke tests.",
    )
    parser.add_argument(
        "--max_test_years",
        type=int,
        default=None,
        help="Optional cap on the number of recursive test years to run.",
    )
    parser.add_argument(
        "--weight_col",
        type=str,
        default=None,
        help=(
            "Optional positive weight column for value-weighted decile backtests. "
            "Leave empty to run equal-weighted deciles."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store predictions and summary files.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print(f"Starting GKX recursive experiment: {args.model}")
    print("=" * 72)
    print(
        f"Hyperparameters -> epochs={args.epochs}, batch_size={args.batch_size:,}, "
        f"lr={args.learning_rate}"
    )

    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = generate_gkx_splits(
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        validation_years=args.validation_years,
    )
    if args.max_test_years is not None:
        splits = splits[: args.max_test_years]

    all_predictions = []
    split_results = []

    for split in splits:
        print("\n" + "-" * 72)
        print(f"Test year {split.test_year}")
        print(
            f"Train: {split.train_start}-{split.train_end} | "
            f"Val: {split.val_start}-{split.val_end} | "
            f"Test: {split.test_start}-{split.test_end}"
        )

        train_generator = GKXDataGenerator(
            filepath=args.data_path,
            batch_size=args.batch_size,
            date_start=split.train_start,
            date_end=split.train_end,
        )
        val_generator = GKXDataGenerator(
            filepath=args.data_path,
            batch_size=args.batch_size,
            date_start=split.val_start,
            date_end=split.val_end,
        )
        test_generator = GKXDataGenerator(
            filepath=args.data_path,
            batch_size=args.batch_size,
            date_start=split.test_start,
            date_end=split.test_end,
            return_metadata=True,
            metadata_cols=_metadata_cols(args.weight_col),
        )

        model = build_neural_net(
            architecture=args.model,
            input_features=train_generator.num_features,
        )

        train_result = train_model(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )

        predictions = predict_model(
            model=train_result["model"],
            generator=test_generator,
        )
        predictions["test_year"] = split.test_year
        all_predictions.append(predictions)

        split_eval = summarize_prediction_panel(predictions)
        split_results.append(
            {
                "test_year": split.test_year,
                "best_val_loss": train_result["best_metric"],
                "test_oos_r2": split_eval["overall_oos_r2"],
            }
        )
        print(
            f"Test year {split.test_year} complete | "
            f"best val loss {train_result['best_metric']:.6f} | "
            f"test OOS R^2 {split_eval['overall_oos_r2']:.6f}"
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_prediction_panel(predictions_df)

    if args.weight_col:
        portfolio_returns, long_short = build_long_short_deciles(
            predictions_df=predictions_df,
            weight_col=args.weight_col,
        )
    else:
        portfolio_returns, long_short = build_long_short_deciles(
            predictions_df=predictions_df,
            weight_col=None,
        )

    split_results_df = pd.DataFrame(split_results)
    long_short_sharpe = annualized_sharpe_ratio(long_short["long_short_10_1"])

    predictions_path = output_dir / f"{args.model.lower()}_predictions.parquet"
    split_results_path = output_dir / f"{args.model.lower()}_split_results.csv"
    monthly_r2_path = output_dir / f"{args.model.lower()}_monthly_oos_r2.csv"
    annual_r2_path = output_dir / f"{args.model.lower()}_annual_oos_r2.csv"
    portfolios_path = output_dir / f"{args.model.lower()}_decile_returns.csv"
    long_short_path = output_dir / f"{args.model.lower()}_long_short.csv"
    summary_path = output_dir / f"{args.model.lower()}_summary.json"

    predictions_df.to_parquet(predictions_path, index=False)
    split_results_df.to_csv(split_results_path, index=False)
    summary["monthly_oos_r2"].to_csv(monthly_r2_path, index=False)
    summary["annual_oos_r2"].to_csv(annual_r2_path, index=False)
    portfolio_returns.to_csv(portfolios_path, index=False)
    long_short.to_csv(long_short_path, index=False)

    summary_payload = {
        "model": args.model,
        "overall_oos_r2": float(summary["overall_oos_r2"]),
        "long_short_sharpe": float(long_short_sharpe),
        "weight_col": args.weight_col,
        "test_start_year": args.test_start_year,
        "test_end_year": splits[-1].test_year,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    elapsed_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 72)
    print(f"Experiment complete in {elapsed_minutes:.2f} minutes")
    print(f"Overall OOS R^2: {summary['overall_oos_r2']:.6f}")
    print(f"Long-short decile Sharpe: {long_short_sharpe:.6f}")
    print(f"Predictions saved to {predictions_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

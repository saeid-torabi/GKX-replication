import argparse
import json
import time
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

from backtest import annualized_sharpe_ratio, build_long_short_deciles
from data_generator import GKXDataGenerator
from evaluate import summarize_prediction_panel
from models import build_neural_net
from splits import generate_gkx_splits
from train import predict_model, train_model


def _metadata_cols(weight_col):
    cols = ["permno", "YYYYMM"]
    if weight_col:
        cols.append(weight_col)
    return cols


def _plot_wealth_growth(long_short, output_dir):
    wealth_df = long_short.copy()
    wealth_df["date"] = pd.to_datetime(
        wealth_df["YYYYMM"].astype(str),
        format="%Y%m",
    )
    wealth_df["wealth"] = (1 + wealth_df["long_short_10_1"]).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(wealth_df["date"], wealth_df["wealth"], linewidth=2)
    plt.title("Cumulative Wealth Growth")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)

    output_path = output_dir / "wealth_growth.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def _plot_learning_curves(all_histories, model_name, output_dir):
    if not all_histories:
        return None

    n_plots = len(all_histories)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, split_history in zip(axes, all_histories):
        history_df = pd.DataFrame(split_history["history"])
        ax.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", linewidth=2)

        if history_df["val_loss"].notna().any():
            ax.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", linewidth=2)

        ax.set_title(f"Test Year {split_history['test_year']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(f"{model_name} Learning Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / f"{model_name.lower()}_learning_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


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
        "--early_stopping_patience",
        type=int,
        default=5,
        help=(
            "Stop training if validation loss fails to improve for this many "
            "consecutive epochs."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help=(
            "Minimum drop in validation loss required to reset the early "
            "stopping patience counter."
        ),
    )
    parser.add_argument(
        "--log_diagnostics",
        action="store_true",
        help=(
            "Log gradient norms plus validation prediction/target summary "
            "statistics for diagnosing unstable NN training."
        ),
    )
    parser.add_argument(
        "--no_shuffle_train",
        action="store_true",
        help="Disable shuffle-buffer randomization for training batches.",
    )
    parser.add_argument(
        "--shuffle_buffer_batches",
        type=int,
        default=8,
        help="Number of streamed batches to hold before shuffling training rows.",
    )
    parser.add_argument(
        "--max_test_years",
        type=int,
        default=None,
        help="Optional cap on the number of recursive test years to run.",
    )
    parser.add_argument(
        "--decile_weight_col",
        type=str,
        default="market_cap",
        help=(
            "Column used for value-weighted stock-level decile portfolios. "
            "Set to an empty string to disable weighting and run equal-weighted deciles."
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
        f"lr={args.learning_rate}, "
        f"early_stopping_patience={args.early_stopping_patience}, "
        f"early_stopping_min_delta={args.early_stopping_min_delta}, "
        f"shuffle_train={not args.no_shuffle_train}, "
        f"shuffle_buffer_batches={args.shuffle_buffer_batches}"
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
    all_histories = []

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
            shuffle=not args.no_shuffle_train,
            shuffle_buffer_batches=args.shuffle_buffer_batches,
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
            metadata_cols=_metadata_cols(args.decile_weight_col or None),
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
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            log_diagnostics=args.log_diagnostics,
        )

        predictions = predict_model(
            model=train_result["model"],
            generator=test_generator,
        )
        predictions["test_year"] = split.test_year
        all_predictions.append(predictions)

        split_eval = summarize_prediction_panel(predictions)
        all_histories.append(
            {
                "test_year": split.test_year,
                "history": train_result["history"],
            }
        )
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
    decile_weight_col = args.decile_weight_col or None
    portfolio_returns, long_short = build_long_short_deciles(
        predictions_df=predictions_df,
        weight_col=decile_weight_col,
    )

    split_results_df = pd.DataFrame(split_results)
    decile_sharpe = annualized_sharpe_ratio(long_short["long_short_10_1"])
    wealth_plot_path = _plot_wealth_growth(long_short, output_dir)
    learning_curves_path = _plot_learning_curves(all_histories, args.model, output_dir)

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
        "decile_strategy_annualized_sharpe": float(decile_sharpe),
        "decile_weight_col": decile_weight_col,
        "test_start_year": args.test_start_year,
        "test_end_year": splits[-1].test_year,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    elapsed_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 72)
    print(f"Experiment complete in {elapsed_minutes:.2f} minutes")
    print(f"Overall OOS R^2: {summary['overall_oos_r2']:.6f}")
    print(f"Decile strategy annualized Sharpe: {decile_sharpe:.6f}")
    print(f"Predictions saved to {predictions_path}")
    print(f"Long-short decile returns saved to {long_short_path}")
    print(f"Wealth growth plot saved to {wealth_plot_path}")
    if learning_curves_path is not None:
        print(f"Learning curves saved to {learning_curves_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

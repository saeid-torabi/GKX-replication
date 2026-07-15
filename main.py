import argparse
import gc
from itertools import product
import json
import random
import shutil
import time
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:  # pragma: no cover - raised later by training code
    torch = None

from backtest import build_long_short_deciles, summarize_decile_performance
from data_generator import GKXDataGenerator
from evaluate import summarize_prediction_panel
from models import build_neural_net
from splits import generate_gkx_splits
from train import predict_model, predict_values, train_model


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _clear_accelerator_cache():
    gc.collect()
    if torch is None:
        return
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_member_results(member_results):
    if member_results is None:
        return
    for result in member_results:
        result.pop("model", None)
    _clear_accelerator_cache()


def _parse_float_list(raw_value, arg_name):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise ValueError(
                f"{arg_name} must be a comma-separated list of numbers. "
                f"Could not parse '{item}'."
            ) from exc

    if not values:
        raise ValueError(f"{arg_name} must contain at least one numeric value.")
    return values


def _average_ensemble_predictions(member_predictions):
    if len(member_predictions) == 1:
        return member_predictions[0]

    base = member_predictions[0].copy()
    prediction_arrays = []
    key_cols = ["permno", "YYYYMM", "excess_ret"]

    for member_idx, predictions in enumerate(member_predictions, start=1):
        if len(predictions) != len(base):
            raise ValueError(
                "Ensemble prediction rows are not aligned: "
                f"member 1 has {len(base):,} rows but member {member_idx} "
                f"has {len(predictions):,} rows."
            )
        for col in key_cols:
            if col in base.columns and col in predictions.columns:
                if not base[col].reset_index(drop=True).equals(
                    predictions[col].reset_index(drop=True)
                ):
                    raise ValueError(
                        "Ensemble prediction rows are not aligned on "
                        f"column '{col}' for member {member_idx}."
                    )
        prediction_arrays.append(predictions["prediction"].to_numpy())

    base["prediction"] = np.mean(np.vstack(prediction_arrays), axis=0)
    return base


def _average_member_histories(member_histories):
    if len(member_histories) == 1:
        return member_histories[0]["history"]

    history_frames = []
    for member_history in member_histories:
        frame = pd.DataFrame(member_history["history"])
        frame["ensemble_member"] = member_history["ensemble_member"]
        frame["seed"] = member_history["seed"]
        history_frames.append(frame)

    combined = pd.concat(history_frames, ignore_index=True)
    metric_cols = [
        col
        for col in [
            "train_loss",
            "train_objective",
            "l1_penalty",
            "val_loss",
            "selection_metric",
            "best_metric",
            "best_epoch",
            "patience_counter",
        ]
        if col in combined.columns
    ]
    averaged = combined.groupby("epoch", as_index=False)[metric_cols].mean()
    return averaged.to_dict(orient="records")


def _member_val_losses(member_results):
    return [
        result["best_metric"]
        for result in member_results
        if result["best_metric"] is not None
    ]


def _mean_result_field(member_results, field):
    values = [result[field] for result in member_results if result.get(field) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _flatten_histories(all_histories):
    rows = []
    for split_history in all_histories:
        for row in split_history["history"]:
            output_row = {"test_year": split_history["test_year"]}
            output_row.update(row)
            rows.append(output_row)
    return pd.DataFrame(rows)


def _resolve_checkpoint_dir(output_dir, checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = output_dir / checkpoint_path
    return checkpoint_path


def _progress_path(checkpoint_dir):
    return checkpoint_dir / "progress.json"


def _read_progress(checkpoint_dir):
    path = _progress_path(checkpoint_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_progress(checkpoint_dir, completed_years, total_years, is_complete=False):
    completed_years = sorted(int(year) for year in completed_years)
    next_year = None
    if completed_years and not is_complete:
        next_year = completed_years[-1] + 1
    payload = {
        "completed_test_years": completed_years,
        "last_completed_test_year": completed_years[-1] if completed_years else None,
        "next_test_year": next_year,
        "completed_year_count": len(completed_years),
        "total_year_count": total_years,
        "is_complete": is_complete,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _progress_path(checkpoint_dir).write_text(json.dumps(payload, indent=2))
    return payload


def _prediction_checkpoint_path(checkpoint_dir, model_name, test_year):
    return checkpoint_dir / f"{model_name.lower()}_predictions_{test_year}.parquet"


def _checkpoint_table_path(checkpoint_dir, model_name, table_name):
    return checkpoint_dir / f"{model_name.lower()}_{table_name}.csv"


def _append_checkpoint_rows(path, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _save_year_checkpoint(
    checkpoint_dir,
    model_name,
    test_year,
    predictions,
    split_result,
    tuning_rows,
    learning_history,
    completed_years,
    total_years,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Make re-saving a year idempotent: drop any stale rows/predictions for it
    # first (guards against a crash between appends and the progress write).
    for table_name in ("split_results", "tuning_results", "learning_history"):
        _remove_year_rows(
            _checkpoint_table_path(checkpoint_dir, model_name, table_name),
            test_year,
        )
    predictions.to_parquet(
        _prediction_checkpoint_path(checkpoint_dir, model_name, test_year),
        index=False,
    )
    _append_checkpoint_rows(
        _checkpoint_table_path(checkpoint_dir, model_name, "split_results"),
        [split_result],
    )
    _append_checkpoint_rows(
        _checkpoint_table_path(checkpoint_dir, model_name, "tuning_results"),
        tuning_rows,
    )
    _append_checkpoint_rows(
        _checkpoint_table_path(checkpoint_dir, model_name, "learning_history"),
        learning_history,
    )
    return _write_progress(
        checkpoint_dir=checkpoint_dir,
        completed_years=completed_years,
        total_years=total_years,
        is_complete=False,
    )


def _checkpoint_prediction_files(checkpoint_dir, model_name):
    return sorted(checkpoint_dir.glob(f"{model_name.lower()}_predictions_*.parquet"))


def _completed_years_from_checkpoints(checkpoint_dir, model_name):
    progress = _read_progress(checkpoint_dir)
    if progress is not None:
        return set(int(year) for year in progress.get("completed_test_years", []))

    completed = set()
    for path in _checkpoint_prediction_files(checkpoint_dir, model_name):
        stem = path.stem
        try:
            completed.add(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return completed


def _read_checkpoint_predictions(checkpoint_dir, model_name):
    prediction_files = _checkpoint_prediction_files(checkpoint_dir, model_name)
    if not prediction_files:
        return None
    return pd.concat(
        [pd.read_parquet(path) for path in prediction_files],
        ignore_index=True,
    )


def _read_checkpoint_table(checkpoint_dir, model_name, table_name):
    path = _checkpoint_table_path(checkpoint_dir, model_name, table_name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _histories_from_frame(history_frame):
    if history_frame.empty:
        return []
    histories = []
    for test_year, frame in history_frame.groupby("test_year", sort=True):
        histories.append(
            {
                "test_year": int(test_year),
                "history": frame.drop(columns=["test_year"]).to_dict(orient="records"),
            }
        )
    return histories


def _remove_year_rows(path, test_year):
    """Drop any existing rows for ``test_year`` so re-saving a year is idempotent."""
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "test_year" in frame.columns:
        frame = frame[frame["test_year"].astype(int) != int(test_year)]
        frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Run-config guard: identity of a run so a resume cannot silently mix settings.
# ---------------------------------------------------------------------------
def _run_config_path(checkpoint_dir):
    return checkpoint_dir / "run_config.json"


def _config_identity(args, tune_learning_rates, tune_l1_lambdas):
    return {
        "model": args.model,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": float(args.learning_rate),
        "l1_lambda": float(args.l1_lambda),
        "tune_hyperparameters": bool(args.tune_hyperparameters),
        "tune_learning_rates": [float(x) for x in tune_learning_rates],
        "tune_l1_lambdas": [float(x) for x in tune_l1_lambdas],
        "full_ensemble_grid": bool(args.full_ensemble_grid),
        "batchnorm_before_relu": bool(args.batchnorm_before_relu),
        "ensemble_size": args.ensemble_size,
        "seed": args.seed,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "validation_years": args.validation_years,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "no_shuffle_train": bool(args.no_shuffle_train),
        "shuffle_buffer_batches": args.shuffle_buffer_batches,
        "max_test_years": args.max_test_years,
        "decile_weight_col": args.decile_weight_col,
    }


def _write_run_config(checkpoint_dir, identity, input_features=None):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(identity)
    existing = _read_run_config(checkpoint_dir)
    if input_features is None and existing is not None:
        input_features = existing.get("input_features")
    if input_features is not None:
        payload["input_features"] = int(input_features)
    _run_config_path(checkpoint_dir).write_text(json.dumps(payload, indent=2))


def _read_run_config(checkpoint_dir):
    path = _run_config_path(checkpoint_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _config_differences(saved_config, identity):
    diffs = []
    for key, value in identity.items():
        if key in saved_config and saved_config[key] != value:
            diffs.append((key, saved_config[key], value))
    return diffs


# ---------------------------------------------------------------------------
# Within-year checkpoints: one file per ensemble member, one marker per combo.
# ---------------------------------------------------------------------------
def _partial_year_dir(checkpoint_dir, test_year):
    return checkpoint_dir / "partial" / f"year_{test_year}"


def _combo_dir(checkpoint_dir, test_year, combo_idx):
    return _partial_year_dir(checkpoint_dir, test_year) / f"combo_{combo_idx:02d}"


def _member_checkpoint_path(combo_dir, member_idx):
    return combo_dir / f"member_{member_idx:02d}.pt"


def _combo_done_path(combo_dir):
    return combo_dir / "combo.json"


def _save_member_checkpoint(
    path,
    train_result,
    member_number,
    seed,
    learning_rate,
    l1_lambda,
    input_features,
    architecture,
    batchnorm_after_relu=True,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": train_result["model"].state_dict(),
        "history": train_result["history"],
        "best_metric": train_result["best_metric"],
        "best_epoch": train_result["best_epoch"],
        "epochs_trained": train_result["epochs_trained"],
        "early_stopped": train_result["early_stopped"],
        "member_number": member_number,
        "seed": seed,
        "learning_rate": learning_rate,
        "l1_lambda": l1_lambda,
        "input_features": int(input_features),
        "architecture": architecture,
        "batchnorm_after_relu": bool(batchnorm_after_relu),
    }
    # Atomic write: a crash mid-write leaves the previous good file (or none),
    # never a truncated checkpoint that would fail to load on resume.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _load_member_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_neural_net(
        architecture=payload["architecture"],
        input_features=payload["input_features"],
        batchnorm_after_relu=payload.get("batchnorm_after_relu", True),
    )
    model.load_state_dict(payload["state_dict"])
    train_result = {
        "model": model,
        "history": payload["history"],
        "best_metric": payload["best_metric"],
        "best_epoch": payload["best_epoch"],
        "epochs_trained": payload["epochs_trained"],
        "early_stopped": payload["early_stopped"],
    }
    member_history = {
        "ensemble_member": payload["member_number"],
        "seed": payload["seed"],
        "history": payload["history"],
    }
    return train_result, member_history


def _train_ensemble_resumable(
    args,
    combo_dir,
    train_generator,
    val_generator,
    input_features,
    learning_rate,
    l1_lambda,
    n_members,
    device=None,
    quiet=False,
):
    """Train ``n_members`` networks, checkpointing after each so a crash resumes
    at the next untrained member instead of restarting the whole ensemble.

    ``n_members`` is decoupled from ``args.ensemble_size`` so grid-search tuning
    can train a single network per candidate while the final selected model is
    trained as the full ensemble."""
    member_results = []
    member_histories = []

    for member_idx in range(n_members):
        member_number = member_idx + 1
        member_seed = args.seed + member_idx
        member_path = _member_checkpoint_path(combo_dir, member_idx)

        if member_path.exists():
            train_result, member_history = _load_member_checkpoint(member_path)
            if not quiet:
                best_metric = train_result["best_metric"]
                best_metric_str = (
                    f"{best_metric:.5f}" if best_metric is not None else "n/a"
                )
                print(
                    f"      net {member_number}/{n_members}  "
                    f"loaded from checkpoint  (val {best_metric_str})"
                )
        else:
            # Print the per-net prefix now (no newline) so the heartbeat dots
            # from train_model stream right after it.
            if not quiet:
                print(
                    f"      net {member_number}/{n_members}  ", end="", flush=True
                )
            _set_global_seed(member_seed)
            model = build_neural_net(
                architecture=args.model,
                input_features=input_features,
                batchnorm_after_relu=not args.batchnorm_before_relu,
            )
            train_result = train_model(
                model=model,
                train_generator=train_generator,
                val_generator=val_generator,
                epochs=args.epochs,
                learning_rate=learning_rate,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                l1_lambda=l1_lambda,
                device=device,
            )
            _save_member_checkpoint(
                path=member_path,
                train_result=train_result,
                member_number=member_number,
                seed=member_seed,
                learning_rate=learning_rate,
                l1_lambda=l1_lambda,
                input_features=input_features,
                architecture=args.model,
                batchnorm_after_relu=not args.batchnorm_before_relu,
            )
            member_history = {
                "ensemble_member": member_number,
                "seed": member_seed,
                "history": train_result["history"],
            }
            if not quiet:
                print(
                    f"  val {train_result['best_metric']:.5f}  "
                    f"(best epoch {train_result['best_epoch']}, "
                    f"{train_result['epochs_trained']} ep)"
                )

        member_results.append(train_result)
        member_histories.append(member_history)

    return member_results, member_histories


def _load_combo_members(args, combo_dir):
    member_results = []
    member_histories = []
    for member_idx in range(args.ensemble_size):
        train_result, member_history = _load_member_checkpoint(
            _member_checkpoint_path(combo_dir, member_idx)
        )
        member_results.append(train_result)
        member_histories.append(member_history)
    return member_results, member_histories


def _run_year_resumable(
    args,
    split,
    checkpoint_dir,
    input_features,
    tune_learning_rates,
    tune_l1_lambdas,
    train_generator,
    val_generator,
    device=None,
):
    """Run one test year's training with combo/member-level checkpointing.

    By default (tune-then-ensemble), the validation grid search trains a single
    network per hyperparameter candidate, and the full ``--ensemble_size``
    ensemble is trained only at the selected candidate. Pass
    ``--full_ensemble_grid`` to instead train the full ensemble at every grid
    point (the older, ~ensemble_size x more expensive behavior).

    Returns the selected ensemble's member results and histories, the selected
    combo, and this year's tuning rows (empty when not tuning)."""
    if args.tune_hyperparameters:
        grid = list(product(tune_learning_rates, tune_l1_lambdas))
    else:
        grid = [(args.learning_rate, args.l1_lambda)]

    # Networks trained per grid candidate. Only the winner gets the full
    # ensemble, unless the full-grid ensemble behavior is explicitly requested.
    if args.tune_hyperparameters and not args.full_ensemble_grid:
        grid_members = 1
    else:
        grid_members = args.ensemble_size

    tuning_rows = []
    best_combo = None

    if args.tune_hyperparameters:
        net_word = "net" if grid_members == 1 else "nets"
        print(f"  tuning: {len(grid)} candidates ({grid_members} {net_word} each)")

    for combo_idx, (candidate_lr, candidate_l1) in enumerate(grid):
        combo_dir = _combo_dir(checkpoint_dir, split.test_year, combo_idx)
        done_path = _combo_done_path(combo_dir)
        tag = f"    [{combo_idx + 1}/{len(grid)}] lr={candidate_lr:g}  l1={candidate_l1:g}"

        if done_path.exists():
            info = json.loads(done_path.read_text())
            candidate_ensemble_loss = info.get("ensemble_val_loss")
            if args.tune_hyperparameters and info.get("tuning_row") is not None:
                tuning_rows.append(info["tuning_row"])
            if args.tune_hyperparameters:
                loss_str = (
                    f"val={candidate_ensemble_loss:.5f}"
                    if candidate_ensemble_loss is not None
                    else "done"
                )
                print(f"{tag}   {loss_str}  (cached)")
        else:
            if args.tune_hyperparameters:
                print(f"{tag}   ", end="", flush=True)
            member_results, _member_histories = _train_ensemble_resumable(
                args=args,
                combo_dir=combo_dir,
                train_generator=train_generator,
                val_generator=val_generator,
                input_features=input_features,
                learning_rate=candidate_lr,
                l1_lambda=candidate_l1,
                n_members=grid_members,
                device=device,
                quiet=args.tune_hyperparameters,
            )

            tuning_row = None
            if args.tune_hyperparameters:
                candidate_losses = _member_val_losses(member_results)
                candidate_ensemble_loss = _ensemble_validation_loss(
                    member_results=member_results,
                    val_generator=val_generator,
                    device=device,
                )
                tuning_row = {
                    "test_year": split.test_year,
                    "learning_rate": candidate_lr,
                    "l1_lambda": candidate_l1,
                    "tuning_nets": grid_members,
                    "ensemble_val_loss": candidate_ensemble_loss,
                    "mean_best_val_loss": float(np.mean(candidate_losses)),
                    "min_best_val_loss": float(np.min(candidate_losses)),
                    "max_best_val_loss": float(np.max(candidate_losses)),
                    "mean_best_epoch": _mean_result_field(
                        member_results, "best_epoch"
                    ),
                    "mean_epochs_trained": _mean_result_field(
                        member_results, "epochs_trained"
                    ),
                    "early_stopped_members": sum(
                        bool(result.get("early_stopped"))
                        for result in member_results
                    ),
                }
                tuning_rows.append(tuning_row)
                print(f"  val={candidate_ensemble_loss:.5f}")
            else:
                candidate_ensemble_loss = None

            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.write_text(
                json.dumps(
                    {
                        "combo_idx": combo_idx,
                        "learning_rate": candidate_lr,
                        "l1_lambda": candidate_l1,
                        "ensemble_val_loss": candidate_ensemble_loss,
                        "tuning_row": tuning_row,
                    },
                    indent=2,
                )
            )
            # Free member models from memory; the selected combo is reloaded
            # from disk once the grid is finished.
            _release_member_results(member_results)

        if args.tune_hyperparameters:
            is_better = (
                best_combo is None or candidate_ensemble_loss < best_combo["loss"]
            )
        else:
            is_better = best_combo is None
        if is_better:
            best_combo = {
                "combo_idx": combo_idx,
                "learning_rate": candidate_lr,
                "l1_lambda": candidate_l1,
                "loss": candidate_ensemble_loss,
            }

    if args.tune_hyperparameters:
        print(
            f"  selected: lr={best_combo['learning_rate']:g}  "
            f"l1={best_combo['l1_lambda']:g}   (val={best_combo['loss']:.5f})"
        )

    best_combo_dir = _combo_dir(checkpoint_dir, split.test_year, best_combo["combo_idx"])

    # Train the full ensemble at the selected configuration. When the grid only
    # trained a single net per candidate (the default), this "tops up" the
    # winning combo from 1 to ensemble_size members. Member 0 already exists and
    # is reused, so only the extra members are trained; member seeds are
    # identical, so the outcome matches training the full ensemble there.
    if args.ensemble_size > grid_members:
        print(f"  ensemble: training {args.ensemble_size} nets at selected config")
        _train_ensemble_resumable(
            args=args,
            combo_dir=best_combo_dir,
            train_generator=train_generator,
            val_generator=val_generator,
            input_features=input_features,
            learning_rate=best_combo["learning_rate"],
            l1_lambda=best_combo["l1_lambda"],
            n_members=args.ensemble_size,
            device=device,
            quiet=False,
        )

    member_results, member_histories = _load_combo_members(args, best_combo_dir)
    return member_results, member_histories, best_combo, tuning_rows


def _write_outputs(
    args,
    output_dir,
    predictions_df,
    split_results_df,
    tuning_results_df,
    learning_history_df,
    elapsed_minutes,
    input_features,
    checkpoint_dir,
    is_complete,
):
    summary = summarize_prediction_panel(predictions_df)
    decile_weight_col = args.decile_weight_col or None
    portfolio_returns, long_short = build_long_short_deciles(
        predictions_df=predictions_df,
        weight_col=decile_weight_col,
    )
    portfolio_performance = summarize_decile_performance(portfolio_returns)
    h_l_row = portfolio_performance.loc[
        portfolio_performance["portfolio"] == "10-1 H-L"
    ].iloc[0]

    all_histories = _histories_from_frame(learning_history_df)
    wealth_plot_path = _plot_wealth_growth(long_short, output_dir)
    learning_curves_path = _plot_learning_curves(
        all_histories,
        args.model,
        output_dir,
    )

    predictions_path = output_dir / f"{args.model.lower()}_predictions.parquet"
    split_results_path = output_dir / f"{args.model.lower()}_split_results.csv"
    monthly_r2_path = output_dir / f"{args.model.lower()}_monthly_oos_r2.csv"
    annual_r2_path = output_dir / f"{args.model.lower()}_annual_oos_r2.csv"
    portfolios_path = output_dir / f"{args.model.lower()}_decile_returns.csv"
    portfolio_performance_path = (
        output_dir / f"{args.model.lower()}_portfolio_performance.csv"
    )
    long_short_path = output_dir / f"{args.model.lower()}_long_short.csv"
    tuning_results_path = output_dir / f"{args.model.lower()}_tuning_results.csv"
    learning_history_path = output_dir / f"{args.model.lower()}_learning_history.csv"
    summary_path = output_dir / f"{args.model.lower()}_summary.json"

    predictions_df.to_parquet(predictions_path, index=False)
    split_results_df.to_csv(split_results_path, index=False)
    summary["monthly_oos_r2"].to_csv(monthly_r2_path, index=False)
    summary["annual_oos_r2"].to_csv(annual_r2_path, index=False)
    portfolio_returns.to_csv(portfolios_path, index=False)
    portfolio_performance.to_csv(portfolio_performance_path, index=False)
    long_short.to_csv(long_short_path, index=False)
    if not tuning_results_df.empty:
        tuning_results_df.to_csv(tuning_results_path, index=False)
    if not learning_history_df.empty:
        learning_history_df.to_csv(learning_history_path, index=False)

    completed_years = sorted(split_results_df["test_year"].astype(int).tolist())
    stock_r2 = float(summary["overall_oos_r2"])
    summary_payload = {
        "model_specifics": {
            "model": args.model,
            "data_path": args.data_path,
            "input_features": int(input_features),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "l1_lambda": float(args.l1_lambda),
            "tune_hyperparameters": args.tune_hyperparameters,
            "tune_learning_rates": _parse_float_list(
                args.tune_learning_rates,
                "--tune_learning_rates",
            ),
            "tune_l1_lambdas": _parse_float_list(
                args.tune_l1_lambdas,
                "--tune_l1_lambdas",
            ),
            "hyperparameter_selection_metric": (
                "ensemble_averaged_validation_mse"
                if args.tune_hyperparameters
                else None
            ),
            "ensemble_size": args.ensemble_size,
            "base_seed": args.seed,
            "ensemble_seed_rule": "member_seed = base_seed + member_index_zero_based",
            "prediction_aggregation": "simple_average_across_ensemble_members",
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "shuffle_train": not args.no_shuffle_train,
            "shuffle_buffer_batches": args.shuffle_buffer_batches,
            "validation_years": args.validation_years,
            "test_start_year": completed_years[0],
            "test_end_year": completed_years[-1],
            "n_test_years": len(completed_years),
            "requested_test_start_year": args.test_start_year,
            "requested_test_end_year": args.test_end_year,
            "is_complete": is_complete,
            "runtime_minutes": elapsed_minutes,
            "checkpoint_dir": str(checkpoint_dir),
        },
        "stock_prediction_performance": {
            "Monthly OOS stock-level prediction performance (R^2)": stock_r2,
            "Monthly OOS stock-level prediction performance (Percentage R^2)": (
                stock_r2 * 100
            ),
            "monthly_oos_r2_csv": str(monthly_r2_path),
            "annual_oos_r2_csv": str(annual_r2_path),
            "split_results_csv": str(split_results_path),
            "learning_history_csv": str(learning_history_path),
            "tuning_results_csv": (
                str(tuning_results_path) if not tuning_results_df.empty else None
            ),
        },
        "machine_learning_portfolios": {
            "weight_col": decile_weight_col,
            "10_minus_1_h_l": _portfolio_row_to_dict(h_l_row),
            "portfolio_performance_csv": str(portfolio_performance_path),
            "monthly_decile_returns_csv": str(portfolios_path),
            "monthly_long_short_csv": str(long_short_path),
        },
        "output_files": {
            "predictions_parquet": str(predictions_path),
            "wealth_growth_png": str(wealth_plot_path),
            "learning_curves_png": (
                str(learning_curves_path) if learning_curves_path else None
            ),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "summary": summary,
        "summary_payload": summary_payload,
        "portfolio_performance": portfolio_performance,
        "h_l_row": h_l_row,
        "paths": {
            "predictions_path": predictions_path,
            "portfolio_performance_path": portfolio_performance_path,
            "long_short_path": long_short_path,
            "wealth_plot_path": wealth_plot_path,
            "learning_curves_path": learning_curves_path,
        },
    }


def _refresh_outputs_from_checkpoints(
    args,
    output_dir,
    checkpoint_dir,
    elapsed_minutes,
    input_features,
    is_complete=False,
):
    predictions_df = _read_checkpoint_predictions(checkpoint_dir, args.model)
    if predictions_df is None:
        return None
    split_results_df = _read_checkpoint_table(
        checkpoint_dir,
        args.model,
        "split_results",
    )
    tuning_results_df = _read_checkpoint_table(
        checkpoint_dir,
        args.model,
        "tuning_results",
    )
    learning_history_df = _read_checkpoint_table(
        checkpoint_dir,
        args.model,
        "learning_history",
    )
    return _write_outputs(
        args=args,
        output_dir=output_dir,
        predictions_df=predictions_df,
        split_results_df=split_results_df,
        tuning_results_df=tuning_results_df,
        learning_history_df=learning_history_df,
        elapsed_minutes=elapsed_minutes,
        input_features=input_features,
        checkpoint_dir=checkpoint_dir,
        is_complete=is_complete,
    )


def _predict_ensemble(member_results, test_generator, device=None):
    member_predictions = []
    for train_result in member_results:
        predictions = predict_model(
            model=train_result["model"],
            generator=test_generator,
            device=device,
        )
        member_predictions.append(predictions)
    return _average_ensemble_predictions(member_predictions)


def _ensemble_validation_loss(member_results, val_generator, device=None):
    prediction_arrays = []
    targets = None

    for member_idx, train_result in enumerate(member_results, start=1):
        member_predictions, member_targets = predict_values(
            model=train_result["model"],
            generator=val_generator,
            device=device,
        )
        if targets is None:
            targets = member_targets
        elif not np.array_equal(targets, member_targets):
            raise ValueError(
                "Ensemble validation rows are not aligned for "
                f"member {member_idx}."
            )
        prediction_arrays.append(member_predictions)

    ensemble_predictions = np.mean(np.vstack(prediction_arrays), axis=0)
    return float(np.mean((ensemble_predictions - targets) ** 2))


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
        has_train_objective = (
            "train_objective" in history_df.columns
            and history_df["train_objective"].notna().any()
        )
        if has_train_objective:
            ax.plot(
                history_df["epoch"],
                history_df["train_objective"],
                label="Train Objective",
                linewidth=2,
            )
            ax.plot(
                history_df["epoch"],
                history_df["train_loss"],
                label="Train MSE",
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
            )
        else:
            ax.plot(
                history_df["epoch"],
                history_df["train_loss"],
                label="Train MSE",
                linewidth=2,
            )

        if history_df["val_loss"].notna().any():
            ax.plot(
                history_df["epoch"],
                history_df["val_loss"],
                label="Val MSE",
                linewidth=2,
            )

        ax.set_title(f"Test Year {split_history['test_year']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Objective / MSE")
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


def _portfolio_row_to_dict(row):
    return {
        "prediction_pct_per_month": float(row["prediction_pct_per_month"]),
        "average_pct_per_month": float(row["average_pct_per_month"]),
        "sd_pct_per_month": float(row["sd_pct_per_month"]),
        "annualized_sharpe_ratio": float(row["annualized_sharpe_ratio"]),
        "n_months": int(row["n_months"]),
    }


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
        "--l1_lambda",
        type=float,
        default=1e-5,
        help=(
            "L1 regularization coefficient for neural-net weight tensors. "
            "Set to 0 to disable; GKX tune small positive values."
        ),
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help=(
            "Run split-level grid search on the rolling validation window. "
            "The selected hyperparameters are then used to score that split's "
            "test year."
        ),
    )
    parser.add_argument(
        "--tune_learning_rates",
        type=str,
        default="0.001,0.01",
        help=(
            "Comma-separated learning-rate grid used when "
            "--tune_hyperparameters is set."
        ),
    )
    parser.add_argument(
        "--tune_l1_lambdas",
        type=str,
        default="1e-5,3e-5,1e-4,3e-4,1e-3",
        help=(
            "Comma-separated L1-penalty grid used when "
            "--tune_hyperparameters is set."
        ),
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help=(
            "Number of independently initialized neural networks to train per "
            "recursive split. Predictions are averaged across members, following "
            "the GKX ensemble approach."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base random seed. Ensemble member k uses seed + k, with k starting "
            "at 0."
        ),
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
        help=(
            "Directory to store predictions and summary files. Resumable "
            "checkpoints are written to a 'checkpoints' subfolder here, and a "
            "run auto-resumes from them if the configuration matches."
        ),
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help=(
            "Delete any existing checkpoints for this run before starting, so "
            "training begins again from the first test year."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help=(
            "Compute device. 'auto' picks mps>cuda>cpu. On Apple silicon, NN1 is "
            "small enough that 'cpu' is often faster than 'mps' (which is memory "
            "bound on 8 GB machines); prefer '--device cpu' there and 'cuda' on a "
            "GPU box."
        ),
    )
    parser.add_argument(
        "--full_ensemble_grid",
        action="store_true",
        help=(
            "Train the full --ensemble_size ensemble at EVERY hyperparameter grid "
            "point (older, ~ensemble_size x more expensive). Default is "
            "tune-then-ensemble: tune with single networks, then train the full "
            "ensemble only at the selected configuration."
        ),
    )
    parser.add_argument(
        "--batchnorm_before_relu",
        action="store_true",
        help=(
            "Place batch normalization BEFORE the ReLU (Linear->BN->ReLU). The "
            "default follows GKX Internet Appendix B.3, which applies batch "
            "normalization AFTER the ReLU (Linear->ReLU->BN)."
        ),
    )

    args = parser.parse_args()
    if args.l1_lambda < 0:
        raise ValueError("--l1_lambda must be non-negative.")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be at least 1.")
    tune_learning_rates = _parse_float_list(
        args.tune_learning_rates,
        "--tune_learning_rates",
    )
    tune_l1_lambdas = _parse_float_list(args.tune_l1_lambdas, "--tune_l1_lambdas")
    if any(value <= 0 for value in tune_learning_rates):
        raise ValueError("--tune_learning_rates values must be positive.")
    if any(value < 0 for value in tune_l1_lambdas):
        raise ValueError("--tune_l1_lambdas values must be non-negative.")

    resolved_device = None if args.device == "auto" else args.device
    bn_order = "BN-before-ReLU" if args.batchnorm_before_relu else "BN-after-ReLU"
    if args.tune_hyperparameters:
        tune_mode = (
            "full-grid ensemble" if args.full_ensemble_grid else "tune-then-ensemble"
        )
    else:
        tune_mode = "fixed hyperparameters"

    print("=" * 72)
    print(
        f"GKX {args.model} — recursive experiment "
        f"{args.test_start_year}-{args.test_end_year}"
    )
    print("=" * 72)
    print(
        f"  device={args.device}  ·  epochs={args.epochs}  ·  "
        f"batch={args.batch_size:,}  ·  ensemble={args.ensemble_size}  ·  "
        f"seed={args.seed}"
    )
    print(
        f"  early-stop patience={args.early_stopping_patience} "
        f"(min_delta={args.early_stopping_min_delta:g})  ·  {bn_order}  ·  "
        f"{tune_mode}"
    )
    if args.tune_hyperparameters:
        print(f"  grid: lr {tune_learning_rates}  x  l1 {tune_l1_lambdas}")

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

    checkpoint_dir = _resolve_checkpoint_dir(output_dir, "checkpoints")
    identity = _config_identity(args, tune_learning_rates, tune_l1_lambdas)

    if args.force_restart and checkpoint_dir.exists():
        print(f"--force_restart set: removing existing checkpoints at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    saved_config = _read_run_config(checkpoint_dir)
    if saved_config is not None:
        diffs = _config_differences(saved_config, identity)
        if diffs:
            diff_lines = "\n".join(
                f"    {key}: checkpoint={old!r} vs requested={new!r}"
                for key, old, new in diffs
            )
            raise SystemExit(
                "Refusing to resume: the requested configuration differs from "
                f"the checkpointed run at {checkpoint_dir}:\n{diff_lines}\n"
                "Re-run with --force_restart to discard those checkpoints and "
                "start over, or choose a new --output_dir."
            )
    _write_run_config(checkpoint_dir, identity)

    completed_years = _completed_years_from_checkpoints(checkpoint_dir, args.model)
    total_years = len(splits)
    if completed_years:
        print(
            f"Resuming from checkpoints: {len(completed_years)} of {total_years} "
            f"test year(s) already complete -> {sorted(completed_years)}"
        )
    input_features = (saved_config or {}).get("input_features")

    for split_idx, split in enumerate(splits, start=1):
        if split.test_year in completed_years:
            print(
                f"[{split_idx}/{total_years}] test year {split.test_year} "
                "already complete, skipping"
            )
            continue

        print("\n" + "-" * 72)
        print(f"[{split_idx}/{total_years}] Test year {split.test_year}")
        print(
            f"  windows: train {split.train_start}-{split.train_end}  "
            f"val {split.val_start}-{split.val_end}  "
            f"test {split.test_start}-{split.test_end}"
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

        if input_features is None:
            input_features = train_generator.num_features
            _write_run_config(
                checkpoint_dir,
                identity,
                input_features=input_features,
            )

        member_results, member_histories, best_combo, tuning_rows = (
            _run_year_resumable(
                args=args,
                split=split,
                checkpoint_dir=checkpoint_dir,
                input_features=input_features,
                tune_learning_rates=tune_learning_rates,
                tune_l1_lambdas=tune_l1_lambdas,
                train_generator=train_generator,
                val_generator=val_generator,
                device=resolved_device,
            )
        )

        predictions = _predict_ensemble(
            member_results, test_generator, device=resolved_device
        )
        predictions["test_year"] = split.test_year

        split_eval = summarize_prediction_panel(predictions)
        member_val_losses = _member_val_losses(member_results)
        averaged_history = _average_member_histories(member_histories)
        learning_history_rows = []
        for history_entry in averaged_history:
            history_row = {"test_year": split.test_year}
            history_row.update(history_entry)
            learning_history_rows.append(history_row)

        split_result = {
            "test_year": split.test_year,
            "ensemble_size": args.ensemble_size,
            "selected_learning_rate": best_combo["learning_rate"],
            "selected_l1_lambda": best_combo["l1_lambda"],
            "selected_ensemble_val_loss": (
                best_combo["loss"] if args.tune_hyperparameters else None
            ),
            "best_val_loss": float(np.mean(member_val_losses)),
            "best_val_loss_min": float(np.min(member_val_losses)),
            "best_val_loss_max": float(np.max(member_val_losses)),
            "mean_best_epoch": _mean_result_field(member_results, "best_epoch"),
            "mean_epochs_trained": _mean_result_field(
                member_results,
                "epochs_trained",
            ),
            "early_stopped_members": sum(
                bool(result.get("early_stopped")) for result in member_results
            ),
            "test_oos_r2": split_eval["overall_oos_r2"],
        }

        completed_years.add(split.test_year)
        _save_year_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_name=args.model,
            test_year=split.test_year,
            predictions=predictions,
            split_result=split_result,
            tuning_rows=tuning_rows,
            learning_history=learning_history_rows,
            completed_years=completed_years,
            total_years=total_years,
        )

        # This year is safely persisted; drop its per-member checkpoints and
        # release the models from memory.
        partial_dir = _partial_year_dir(checkpoint_dir, split.test_year)
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        _release_member_results(member_results)

        # Rewrite every output file so the newest completed year is always
        # reflected, even if the run is interrupted before finishing.
        elapsed_minutes = (time.time() - start_time) / 60
        _refresh_outputs_from_checkpoints(
            args=args,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            elapsed_minutes=elapsed_minutes,
            input_features=input_features,
            is_complete=False,
        )
        print(
            f"  result: OOS R2 = {split_eval['overall_oos_r2'] * 100:+.4f}%   "
            f"(mean val loss {np.mean(member_val_losses):.5f})  ·  outputs refreshed"
        )

    if input_features is None:
        input_features = (_read_run_config(checkpoint_dir) or {}).get(
            "input_features"
        )
    if input_features is None:
        # Nothing trained this invocation and the value was never stored; derive
        # it cheaply from the first split's training window so outputs can build.
        probe_generator = GKXDataGenerator(
            filepath=args.data_path,
            batch_size=args.batch_size,
            date_start=splits[0].train_start,
            date_end=splits[0].train_end,
        )
        input_features = probe_generator.num_features
        _write_run_config(checkpoint_dir, identity, input_features=input_features)

    _write_progress(
        checkpoint_dir=checkpoint_dir,
        completed_years=completed_years,
        total_years=total_years,
        is_complete=True,
    )
    elapsed_minutes = (time.time() - start_time) / 60
    result = _refresh_outputs_from_checkpoints(
        args=args,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        elapsed_minutes=elapsed_minutes,
        input_features=input_features,
        is_complete=True,
    )
    if result is None:
        raise SystemExit(
            "No completed test years were found, so nothing was written. "
            f"Check the checkpoint directory at {checkpoint_dir}."
        )

    _print_final_summary(
        args=args,
        result=result,
        checkpoint_dir=checkpoint_dir,
        elapsed_minutes=elapsed_minutes,
        tune_learning_rates=tune_learning_rates,
        tune_l1_lambdas=tune_l1_lambdas,
    )


def _print_final_summary(
    args,
    result,
    checkpoint_dir,
    elapsed_minutes,
    tune_learning_rates,
    tune_l1_lambdas,
):
    summary_payload = result["summary_payload"]
    model_specifics = summary_payload["model_specifics"]
    h_l_row = result["h_l_row"]
    paths = result["paths"]
    stock_r2 = float(result["summary"]["overall_oos_r2"])
    decile_weight_col = args.decile_weight_col or None

    print("\n" + "=" * 72)
    print("Experiment Summary")
    print("=" * 72)
    print("Model specifics")
    print(f"  Model: {args.model}")
    print(
        f"  Test window: {model_specifics['test_start_year']}-"
        f"{model_specifics['test_end_year']} "
        f"({model_specifics['n_test_years']} year(s) completed)"
    )
    print(
        f"  Hyperparameters: epochs={args.epochs}, "
        f"batch_size={args.batch_size:,}, lr={args.learning_rate}, "
        f"l1_lambda={args.l1_lambda}"
    )
    if args.tune_hyperparameters:
        print(
            "  Validation tuning: enabled, "
            f"learning_rates={tune_learning_rates}, "
            f"l1_lambdas={tune_l1_lambdas}, "
            "selection=ensemble averaged validation MSE"
        )
    print(f"  Ensemble: size={args.ensemble_size}, base_seed={args.seed}")
    print(
        f"  Early stopping: patience={args.early_stopping_patience}, "
        f"min_delta={args.early_stopping_min_delta}"
    )
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Runtime this session: {elapsed_minutes:.2f} minutes")
    print("\nStock prediction performance")
    print(
        "  Monthly OOS stock-level prediction performance "
        f"(Percentage R^2): {stock_r2 * 100:.4f}%"
    )
    print("\nPerformance of the machine learning portfolios")
    print(f"  Weighting: {decile_weight_col or 'equal_weight'}")
    print(
        "  10-1 H-L prediction: "
        f"{h_l_row['prediction_pct_per_month']:.4f}% per month"
    )
    print(
        "  10-1 H-L average: "
        f"{h_l_row['average_pct_per_month']:.4f}% per month"
    )
    print(f"  10-1 H-L SD: {h_l_row['sd_pct_per_month']:.4f}% per month")
    print(
        "  10-1 H-L Sharpe Ratio: "
        f"{h_l_row['annualized_sharpe_ratio']:.6f}"
    )
    print("\nSaved outputs")
    print(f"Predictions saved to {paths['predictions_path']}")
    print(f"Portfolio performance saved to {paths['portfolio_performance_path']}")
    print(f"Long-short decile returns saved to {paths['long_short_path']}")
    print(f"Wealth growth plot saved to {paths['wealth_plot_path']}")
    if paths.get("learning_curves_path") is not None:
        print(f"Learning curves saved to {paths['learning_curves_path']}")
    print("=" * 72)


if __name__ == "__main__":
    main()

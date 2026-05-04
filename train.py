import copy
import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised at runtime
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def get_device(device=None):
    if torch is None:
        raise ImportError(
            "torch is required for model training. "
            "Install it with `pip install torch`."
        ) from _TORCH_IMPORT_ERROR

    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
    }


def _format_stats(name, stats):
    return (
        f"{name}[min={stats['min']:.6g}, "
        f"max={stats['max']:.6g}, "
        f"mean={stats['mean']:.6g}, "
        f"std={stats['std']:.6g}]"
    )


def _gradient_norm(model):
    total_sq_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        total_sq_norm += param_norm ** 2
    return total_sq_norm ** 0.5


def evaluate_loss(model, generator, device, max_batches=None, collect_diagnostics=False):
    model.eval()
    # The paper's neural networks minimize a penalized l2 objective.
    loss_fn = torch.nn.MSELoss()
    running_loss = 0.0
    batches_seen = 0
    prediction_chunks = []
    target_chunks = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(generator, start=1):
            x_batch, y_batch = batch[:2]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            running_loss += loss.item()
            batches_seen = batch_idx
            if collect_diagnostics:
                prediction_chunks.append(predictions.detach().cpu().reshape(-1))
                target_chunks.append(y_batch.detach().cpu().reshape(-1))

            if max_batches is not None and batch_idx >= max_batches:
                break

    val_loss = running_loss / batches_seen
    if not collect_diagnostics:
        return val_loss

    prediction_tensor = torch.cat(prediction_chunks)
    target_tensor = torch.cat(target_chunks)
    return {
        "loss": val_loss,
        "prediction_stats": _tensor_stats(prediction_tensor),
        "target_stats": _tensor_stats(target_tensor),
    }


def train_model(
    model,
    train_generator,
    epochs,
    learning_rate=1e-3,
    device=None,
    log_every=50,
    max_train_batches=None,
    val_generator=None,
    max_val_batches=None,
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    log_diagnostics=False,
):
    device = get_device(device)
    print(f"Training on device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # The paper's NN objective is l2, not Huber.
    loss_fn = torch.nn.MSELoss()

    history = []
    best_metric = None
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches_seen = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_generator, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            grad_norm = _gradient_norm(model)
            optimizer.step()

            running_loss += loss.item()
            batches_seen = batch_idx

            if batch_idx % log_every == 0 or batch_idx == len(train_generator):
                avg_so_far = running_loss / batch_idx
                log_parts = [
                    f"epoch={epoch:03d}/{epochs:03d}",
                    f"batch={batch_idx:05d}/{len(train_generator):05d}",
                    f"avg_train_loss={avg_so_far:.6f}",
                ]
                if log_diagnostics:
                    log_parts.append(f"grad_l2={grad_norm:.6g}")
                print(" | ".join(log_parts))

            if max_train_batches is not None and batch_idx >= max_train_batches:
                print(f"⚡️ Stopped early after {batch_idx} training batches for this epoch.")
                break

        train_loss = running_loss / batches_seen

        if val_generator is not None:
            val_result = evaluate_loss(
                model=model,
                generator=val_generator,
                device=device,
                max_batches=max_val_batches,
                collect_diagnostics=log_diagnostics,
            )
            if log_diagnostics:
                val_loss = val_result["loss"]
                prediction_stats = val_result["prediction_stats"]
                target_stats = val_result["target_stats"]
            else:
                val_loss = val_result
            selection_metric = val_loss
            log_parts = [
                f"epoch={epoch:03d} complete",
                f"train_loss={train_loss:.6f}",
                f"val_loss={val_loss:.6f}",
            ]
            if log_diagnostics:
                log_parts.extend(
                    [
                        _format_stats("val_pred", prediction_stats),
                        _format_stats("val_target", target_stats),
                    ]
                )
            print(" | ".join(log_parts))
        else:
            val_loss = None
            selection_metric = train_loss
            print(f"Epoch {epoch:03d} complete | mean training loss {train_loss:.6f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        improved = (
            best_metric is None
            or selection_metric < (best_metric - early_stopping_min_delta)
        )

        if improved:
            best_metric = selection_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            val_generator is not None
            and early_stopping_patience is not None
            and patience_counter >= early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"validation loss failed to improve by more than "
                f"{early_stopping_min_delta:.6f} for "
                f"{early_stopping_patience} consecutive epochs."
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_metric": best_metric,
    }


def predict_model(model, generator, device=None, prediction_col="prediction"):
    device = get_device(device)
    model = model.to(device)
    model.eval()

    outputs = []

    with torch.no_grad():
        for batch in generator:
            x_batch, y_batch, metadata_df = batch
            predictions = model(x_batch.to(device)).cpu().numpy().reshape(-1)
            realized = y_batch.cpu().numpy().reshape(-1)

            batch_output = metadata_df.copy()
            batch_output[prediction_col] = predictions
            batch_output["excess_ret"] = realized
            outputs.append(batch_output)

    return pd.concat(outputs, ignore_index=True)

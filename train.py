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


def evaluate_loss(model, generator, device, max_batches=None):
    model.eval()
    loss_fn = torch.nn.HuberLoss()
    running_loss = 0.0
    batches_seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(generator, start=1):
            x_batch, y_batch = batch[:2]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            running_loss += loss.item()
            batches_seen = batch_idx

            if max_batches is not None and batch_idx >= max_batches:
                break

    return running_loss / batches_seen


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
):
    device = get_device(device)
    print(f"Training on device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.HuberLoss()

    history = []
    best_metric = None
    best_state_dict = None

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
            optimizer.step()

            running_loss += loss.item()
            batches_seen = batch_idx

            if batch_idx % log_every == 0 or batch_idx == len(train_generator):
                avg_so_far = running_loss / batch_idx
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Batch {batch_idx:05d}/{len(train_generator):05d} | "
                    f"Avg train loss {avg_so_far:.6f}"
                )

            if max_train_batches is not None and batch_idx >= max_train_batches:
                print(f"Stopped early after {batch_idx} training batches for this epoch.")
                break

        train_loss = running_loss / batches_seen

        if val_generator is not None:
            val_loss = evaluate_loss(
                model=model,
                generator=val_generator,
                device=device,
                max_batches=max_val_batches,
            )
            selection_metric = val_loss
            print(
                f"Epoch {epoch:03d} complete | "
                f"train loss {train_loss:.6f} | val loss {val_loss:.6f}"
            )
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

        if best_metric is None or selection_metric < best_metric:
            best_metric = selection_metric
            best_state_dict = copy.deepcopy(model.state_dict())

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

import copy
import numpy as np
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


def _l1_regularized_parameters(model):
    """
    Yield weight tensors covered by the GKX-style l1 penalty.

    Biases and one-dimensional batch-normalization parameters are excluded so
    the penalty targets network weights rather than affine offsets.
    """
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim > 1:
            yield name, parameter


def _l1_penalty(model, device):
    penalty = None
    for _, parameter in _l1_regularized_parameters(model):
        parameter_penalty = parameter.abs().sum()
        penalty = parameter_penalty if penalty is None else penalty + parameter_penalty

    if penalty is None:
        return torch.zeros((), device=device)
    return penalty


def evaluate_loss(model, generator, device):
    """Mean predictive MSE over ``generator`` (no regularization)."""
    model.eval()
    loss_fn = torch.nn.MSELoss()
    running_loss = 0.0
    batches_seen = 0

    with torch.no_grad():
        for batch in generator:
            x_batch, y_batch = batch[:2]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            running_loss += loss_fn(predictions, y_batch).item()
            batches_seen += 1

    return running_loss / batches_seen


def train_model(
    model,
    train_generator,
    epochs,
    learning_rate=1e-3,
    device=None,
    val_generator=None,
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    l1_lambda=1e-5,
):
    """Fit one network with Adam on an MSE + L1 objective, keeping the best
    epoch by validation loss and supporting early stopping.

    Prints one flushed '.' per completed epoch as a liveness heartbeat.
    """
    if l1_lambda < 0:
        raise ValueError("l1_lambda must be non-negative.")

    device = get_device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # The paper's NN objective is MSE plus an l1 penalty, not Huber.
    loss_fn = torch.nn.MSELoss()

    history = []
    best_metric = None
    best_state_dict = None
    best_epoch = None
    patience_counter = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        model.train()
        running_mse = 0.0
        running_objective = 0.0
        running_l1_penalty = 0.0
        batches_seen = 0

        for x_batch, y_batch in train_generator:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_batch)
            mse_loss = loss_fn(predictions, y_batch)
            if l1_lambda > 0:
                l1_penalty = _l1_penalty(model, device)
            else:
                l1_penalty = torch.zeros((), device=device)
            loss = mse_loss + (l1_lambda * l1_penalty)
            loss.backward()
            optimizer.step()

            running_mse += mse_loss.item()
            running_objective += loss.item()
            running_l1_penalty += l1_penalty.item()
            batches_seen += 1

        train_loss = running_mse / batches_seen
        train_objective = running_objective / batches_seen
        train_l1_penalty = running_l1_penalty / batches_seen

        if val_generator is not None:
            val_loss = evaluate_loss(model, val_generator, device)
            selection_metric = val_loss
        else:
            val_loss = None
            selection_metric = train_loss

        # Heartbeat: one flushed mark per completed epoch so a long run visibly
        # shows it is alive.
        print(".", end="", flush=True)

        improved = (
            best_metric is None
            or selection_metric < (best_metric - early_stopping_min_delta)
        )

        if improved:
            best_metric = selection_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_objective": train_objective,
                "l1_penalty": train_l1_penalty,
                "val_loss": val_loss,
                "selection_metric": selection_metric,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "improved": improved,
                "patience_counter": patience_counter,
            }
        )

        if (
            val_generator is not None
            and early_stopping_patience is not None
            and patience_counter >= early_stopping_patience
        ):
            early_stopped = True
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "early_stopped": early_stopped,
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


def predict_values(model, generator, device=None):
    device = get_device(device)
    model = model.to(device)
    model.eval()

    prediction_chunks = []
    target_chunks = []

    with torch.no_grad():
        for batch in generator:
            x_batch, y_batch = batch[:2]
            predictions = model(x_batch.to(device)).cpu().numpy().reshape(-1)
            targets = y_batch.cpu().numpy().reshape(-1)
            prediction_chunks.append(predictions)
            target_chunks.append(targets)

    return (
        np.concatenate(prediction_chunks),
        np.concatenate(target_chunks),
    )

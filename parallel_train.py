"""
Vectorized (parallel) ensemble training for GPU runs.

A single GKX network is tiny, so on a GPU it barely occupies the device and
training one-at-a-time is slow. This module trains K networks *concurrently* by
stacking their parameters and running one batched (bmm) forward/backward pass,
with manual per-network batch normalization. It reproduces the sequential
training method faithfully:

  * each network keeps its own initialization (seed = base_seed + member_index,
    built from the real models.py definition, then stacked),
  * each network shuffles its own data with its own RNG stream,
  * each network has its own batch-norm statistics,
  * each network has its own early-stopping / best-epoch selection.

The only differences from a sequential CPU run are unavoidable for any parallel
GPU implementation: the per-network shuffle stream differs from the global-RNG
stream, and batched ops sum in a different floating-point order. So a parallel
run is a *valid, faithful draw* of the ensemble, not a bit-identical copy of the
CPU run.

Correctness safeguard: because this file may be written in an environment with
no torch, ``parallel_self_check`` verifies the vectorized forward AND backward
against reference nn.Module instances before any real training. A mismatch
raises loudly.
"""
import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised at runtime
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc

from models import build_neural_net


# ---------------------------------------------------------------------------
# Introspect a reference network into an ordered list of layer operations, so
# the vectorized forward works for NN1..NN5 and either batch-norm ordering.
# ---------------------------------------------------------------------------
def _describe_layers(ref_model):
    layers = []
    for idx, module in enumerate(ref_model.network):
        if isinstance(module, nn.Linear):
            layers.append(
                {"kind": "linear", "w": f"network.{idx}.weight",
                 "b": f"network.{idx}.bias"}
            )
        elif isinstance(module, nn.ReLU):
            layers.append({"kind": "relu"})
        elif isinstance(module, nn.BatchNorm1d):
            layers.append({
                "kind": "bn",
                "w": f"network.{idx}.weight", "b": f"network.{idx}.bias",
                "rm": f"network.{idx}.running_mean",
                "rv": f"network.{idx}.running_var",
                "eps": module.eps, "momentum": module.momentum,
            })
        else:
            raise ValueError(f"Unsupported layer for vectorization: {type(module)}")
    return layers


def _stack_state(models):
    """Stack K models' parameters and buffers into {name: (K, *shape)} dicts.
    Parameters are leaf tensors requiring grad; buffers are plain tensors."""
    param_names = [n for n, _ in models[0].named_parameters()]
    buffer_names = [n for n, _ in models[0].named_buffers()]
    params, buffers = {}, {}
    for name in param_names:
        stacked = torch.stack([dict(m.named_parameters())[name].detach()
                               for m in models], dim=0)
        params[name] = stacked.clone().requires_grad_(True)
    for name in buffer_names:
        stacked = torch.stack([dict(m.named_buffers())[name].detach()
                               for m in models], dim=0)
        buffers[name] = stacked.clone()
    return params, buffers


def _vectorized_forward(layers, params, buffers, x, training):
    """Batched forward for K nets. x: (K, B, in_features) -> (K, B, 1)."""
    out = x
    for layer in layers:
        if layer["kind"] == "linear":
            w = params[layer["w"]]                       # (K, out, in)
            b = params[layer["b"]]                       # (K, out)
            out = torch.bmm(out, w.transpose(1, 2)) + b.unsqueeze(1)
        elif layer["kind"] == "relu":
            out = torch.relu(out)
        elif layer["kind"] == "bn":
            gamma = params[layer["w"]].unsqueeze(1)      # (K, 1, C)
            beta = params[layer["b"]].unsqueeze(1)
            eps = layer["eps"]
            if training:
                mean = out.mean(dim=1, keepdim=True)                 # (K,1,C)
                var_b = out.var(dim=1, unbiased=False, keepdim=True)  # biased
                normed = (out - mean) / torch.sqrt(var_b + eps)
                with torch.no_grad():
                    var_u = out.var(dim=1, unbiased=True)             # (K,C)
                    m = layer["momentum"]
                    buffers[layer["rm"]].mul_(1 - m).add_(
                        mean.squeeze(1).detach(), alpha=m)
                    buffers[layer["rv"]].mul_(1 - m).add_(
                        var_u.detach(), alpha=m)
            else:
                rm = buffers[layer["rm"]].unsqueeze(1)
                rv = buffers[layer["rv"]].unsqueeze(1)
                normed = (out - rm) / torch.sqrt(rv + eps)
            out = normed * gamma + beta
    return out


def _l1_weight_penalty(params):
    """Per-net L1 over weight tensors only (per-net ndim>1: the Linear weights,
    stacked as (K, out, in)). Returns (K,)."""
    total = None
    for name, tensor in params.items():
        if tensor.ndim == 3:  # stacked 2-D weight -> Linear weight matrix
            term = tensor.abs().sum(dim=(1, 2))
            total = term if total is None else total + term
    return total


def parallel_self_check(architecture, input_features, batchnorm_after_relu,
                        device, k=3, batch=64, tol=1e-4):
    """Verify the vectorized forward+backward matches reference nn.Modules.
    Raises AssertionError on mismatch. Cheap; run once before real training."""
    torch.manual_seed(0)
    models = [build_neural_net(architecture, input_features, batchnorm_after_relu)
              for _ in range(k)]
    for m in models:
        m.to(device).train()
    layers = _describe_layers(models[0])
    params, buffers = _stack_state(models)

    x = torch.randn(k, batch, input_features, device=device)
    y = torch.randn(k, batch, device=device)
    l1_lambda = 1e-3  # exercise the L1-gradient path too

    def ref_l1(model):
        total = None
        for _, p in model.named_parameters():
            if p.ndim > 1:
                term = p.abs().sum()
                total = term if total is None else total + term
        return total

    # Reference: run each model independently (MSE + L1).
    ref_out, ref_losses = [], []
    for i, m in enumerate(models):
        oi = m(x[i])                       # (B,1)
        ref_out.append(oi)
        ref_losses.append(((oi.squeeze(-1) - y[i]) ** 2).mean()
                          + l1_lambda * ref_l1(m))
    ref_out = torch.stack(ref_out, dim=0)  # (K,B,1)
    ref_total = torch.stack(ref_losses).sum()
    ref_total.backward()
    ref_grads = {n: torch.stack([dict(m.named_parameters())[n].grad
                                 for m in models], dim=0)
                 for n, _ in models[0].named_parameters()}

    # Vectorized (MSE + L1).
    vout = _vectorized_forward(layers, params, buffers, x, training=True)
    vloss = (((vout.squeeze(-1) - y) ** 2).mean(dim=1)
             + l1_lambda * _l1_weight_penalty(params)).sum()
    vloss.backward()

    fwd_ok = torch.allclose(vout, ref_out, atol=tol, rtol=tol)
    if not fwd_ok:
        raise AssertionError(
            "parallel_self_check: vectorized FORWARD does not match reference "
            f"modules (max abs diff {(vout - ref_out).abs().max().item():.3e})."
        )
    for name in ref_grads:
        vg = params[name].grad
        if not torch.allclose(vg, ref_grads[name], atol=tol, rtol=tol):
            raise AssertionError(
                f"parallel_self_check: vectorized BACKWARD grad for '{name}' "
                f"does not match reference (max abs diff "
                f"{(vg - ref_grads[name]).abs().max().item():.3e})."
            )
    return True


def train_parallel_members(
    architecture,
    input_features,
    member_specs,          # list of (member_number, seed)
    train_x, train_y,      # tensors on device: (N, F), (N,)
    val_x, val_y,          # tensors on device
    epochs,
    learning_rate,
    l1_lambda,
    early_stopping_patience,
    early_stopping_min_delta,
    device,
    batchnorm_after_relu=True,
    batch_size=10000,
    run_self_check=True,
):
    """Train K networks (same lr/lambda, distinct seeds) concurrently. Returns a
    list of train_result dicts matching train.train_model's output, one per
    member, in the order of member_specs."""
    if torch is None:
        raise ImportError("torch is required for parallel training.") \
            from _TORCH_IMPORT_ERROR

    if run_self_check:
        parallel_self_check(architecture, input_features, batchnorm_after_relu,
                            device)

    k = len(member_specs)
    # Build K reference modules with faithful per-member initialization.
    models = []
    for _, seed in member_specs:
        _seed_everything(seed)
        models.append(
            build_neural_net(architecture, input_features, batchnorm_after_relu)
            .to(device)
        )
    layers = _describe_layers(models[0])
    params, buffers = _stack_state(models)
    optimizer = torch.optim.Adam(list(params.values()), lr=learning_rate)

    n_train = train_x.shape[0]
    generators = []
    for _, seed in member_specs:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        generators.append(g)

    histories = [[] for _ in range(k)]
    best_metric = [None] * k
    best_state = [None] * k
    best_epoch = [None] * k
    patience = [0] * k
    stopped = [False] * k

    def snapshot(i):
        state = {n: params[n][i].detach().cpu().clone() for n in params}
        state.update({n: buffers[n][i].detach().cpu().clone() for n in buffers})
        return state

    for epoch in range(1, epochs + 1):
        # Per-member permutation of the shared training rows.
        perms = torch.stack(
            [torch.randperm(n_train, generator=generators[i], device=device)
             for i in range(k)],
            dim=0,
        )  # (K, N)
        n_batches = (n_train + batch_size - 1) // batch_size
        run_mse = torch.zeros(k, device=device)
        run_obj = torch.zeros(k, device=device)
        run_l1 = torch.zeros(k, device=device)
        seen = 0
        for bi in range(n_batches):
            idx = perms[:, bi * batch_size:(bi + 1) * batch_size]  # (K, b)
            if idx.shape[1] < 2:
                continue
            xb = train_x[idx]                    # (K, b, F)
            yb = train_y[idx]                    # (K, b)
            preds = _vectorized_forward(layers, params, buffers, xb, training=True)
            mse = ((preds.squeeze(-1) - yb) ** 2).mean(dim=1)     # (K,)
            l1 = _l1_weight_penalty(params)                       # (K,)
            per_net = mse + l1_lambda * l1
            # Only still-active members contribute gradients.
            active = torch.tensor([0.0 if s else 1.0 for s in stopped],
                                  device=device, dtype=per_net.dtype)
            loss = (per_net * active).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                run_mse += mse.detach()
                run_obj += per_net.detach()
                run_l1 += l1.detach()
            seen += 1

        denom = max(seen, 1)
        run_mse /= denom
        run_obj /= denom
        run_l1 /= denom

        with torch.no_grad():
            # Batch the validation forward to bound memory: a full (K, N_val, F)
            # tensor would be enormous for a 12-year validation window.
            n_val = val_x.shape[0]
            sse = torch.zeros(k, device=device)
            for vb in range(0, n_val, batch_size):
                vx = val_x[vb:vb + batch_size]                     # (b, F)
                vy = val_y[vb:vb + batch_size]                     # (b,)
                vx_k = vx.unsqueeze(0).expand(k, -1, -1).contiguous()
                vp = _vectorized_forward(layers, params, buffers, vx_k,
                                         training=False)
                sse += ((vp.squeeze(-1) - vy.unsqueeze(0)) ** 2).sum(dim=1)
            val_mse = sse / n_val               # (K,)

        for i in range(k):
            if stopped[i]:
                continue
            vloss = float(val_mse[i].item())
            improved = (best_metric[i] is None
                        or vloss < best_metric[i] - early_stopping_min_delta)
            if improved:
                best_metric[i] = vloss
                best_epoch[i] = epoch
                best_state[i] = snapshot(i)
                patience[i] = 0
            else:
                patience[i] += 1
            histories[i].append({
                "epoch": epoch,
                "train_loss": float(run_mse[i].item()),
                "train_objective": float(run_obj[i].item()),
                "l1_penalty": float(run_l1[i].item()),
                "val_loss": vloss,
                "selection_metric": vloss,
                "best_metric": best_metric[i],
                "best_epoch": best_epoch[i],
                "improved": improved,
                "patience_counter": patience[i],
            })
            if patience[i] >= early_stopping_patience:
                stopped[i] = True
        if all(stopped):
            break

    # Load each member's best state back into its reference module.
    results = []
    for i, (_, seed) in enumerate(member_specs):
        state = best_state[i] if best_state[i] is not None else {
            **{n: params[n][i].detach().cpu().clone() for n in params},
            **{n: buffers[n][i].detach().cpu().clone() for n in buffers},
        }
        models[i].load_state_dict(state)
        results.append({
            "model": models[i],
            "history": histories[i],
            "best_metric": best_metric[i],
            "best_epoch": best_epoch[i],
            "epochs_trained": len(histories[i]),
            "early_stopped": stopped[i],
        })
    return results


def _seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def materialize_dataset(generator, device):
    """Stack all rows of a GKXDataGenerator into device tensors (X, y). Uses the
    ordered iterator (not the shuffle buffer) so every row is included exactly
    once; each member re-shuffles independently during training."""
    if hasattr(generator, "_iter_ordered_batches"):
        iterator = generator._iter_ordered_batches()
    else:
        iterator = iter(generator)
    xs, ys = [], []
    for batch in iterator:
        xs.append(batch[0])
        ys.append(batch[1].reshape(-1))
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y

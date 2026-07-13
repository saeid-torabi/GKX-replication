try:
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised at runtime
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class FeedForwardNN(nn.Module if nn is not None else object):
    def __init__(
        self,
        input_features,
        hidden_layers,
        dropout=0.0,
        batchnorm_after_relu=True,
    ):
        if nn is None:
            raise ImportError(
                "torch is required to build neural network models. "
                "Install it with `pip install torch`."
            ) from _TORCH_IMPORT_ERROR

        super().__init__()

        layers = []
        in_features = input_features

        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            # GKX (Internet Appendix B.3) apply batch normalization to each
            # activation AFTER the ReLU transformation, so the default order is
            # Linear -> ReLU -> BatchNorm. Set batchnorm_after_relu=False for the
            # more common Linear -> BatchNorm -> ReLU ordering.
            if batchnorm_after_relu:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_units))
            else:
                layers.append(nn.BatchNorm1d(hidden_units))
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_units

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_neural_net(architecture, input_features, batchnorm_after_relu=True):
    """
    Build a small family of GKX-style feed-forward networks.

    The paper compares shallow and deeper neural nets; here we expose a few
    practical presets that can be swapped from the CLI. ``batchnorm_after_relu``
    controls whether batch normalization is applied after the ReLU (the GKX
    Internet Appendix ordering, and the default) or before it.
    """
    architecture = architecture.upper()

    configs = {
        "NN1": {"hidden_layers": [32], "dropout": 0.0},
        "NN2": {"hidden_layers": [32, 16], "dropout": 0.0},
        "NN3": {"hidden_layers": [32, 16, 8], "dropout": 0.0},
        "NN4": {"hidden_layers": [32, 16, 8, 4], "dropout": 0.0},
        "NN5": {"hidden_layers": [32, 16, 8, 4, 2], "dropout": 0.0},
    }

    if architecture not in configs:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Expected one of {sorted(configs)}."
        )

    config = configs[architecture]
    bn_order = "after_relu" if batchnorm_after_relu else "before_relu"
    print(
        f"Model config -> architecture={architecture}, "
        f"input_features={input_features}, hidden_layers={config['hidden_layers']}, "
        f"batchnorm={bn_order}"
    )

    return FeedForwardNN(
        input_features=input_features,
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        batchnorm_after_relu=batchnorm_after_relu,
    )

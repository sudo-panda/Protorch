import torch.nn as nn
import torch

"""
MLP (Multi-Layer Perceptron) model class.

Sample config:
[
    { "Linear":     [128] },
    { "ReLU":       []    },
    { "LayerNorm":  [128] },
    { "Dropout":    [0.1] },
    { "Linear":     [256] },
    { "LayerNorm":  [256] },
    { "Dropout":    [0.1] },
    { "Linear":     [512] },
    { "ReLU":       []    },
    { "LayerNorm":  [512] },
    { "Dropout":    [0.1] }
]
"""


class MLP(nn.Module):
    layer_map = {
        "Linear": nn.Linear,
        "ReLU": nn.ReLU,
        "Dropout": nn.Dropout,
        "BatchNorm1d": nn.BatchNorm1d,
        "LayerNorm": nn.LayerNorm,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid
    }

    def __init__(self, layer_list: list[dict[str, list]], input_dim: int):
        super().__init__()

        self.input_dim = input_dim

        layers = []
        current_dim = input_dim

        for layer_spec in layer_list:
            if len(layer_spec) != 1:
                raise ValueError(f"Each layer spec should contain exactly one key (the layer name)" \
                                 f"\nSpec: {layer_spec}\nList: {layer_list}")

            layer_type, args = next(iter(layer_spec.items()))
            layer_cls = self.layer_map.get(layer_type)
            if not layer_cls:
                raise ValueError(f"Unknown layer type: {layer_type}")

            # Handle dimension inference for layers like Linear
            if layer_type == "Linear":
                out_dim = args[0]
                layer = layer_cls(current_dim, out_dim)
                current_dim = out_dim  # Update current shape
            elif layer_type in {"BatchNorm1d", "LayerNorm"}:
                # These also need the current dimension
                layer = layer_cls(current_dim)
            else:
                # No args needed (e.g., ReLU, Tanh, etc.)
                layer = layer_cls(*args)

            layers.append(layer)

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    
    def get_output_dim(self):
        for layer in reversed(self.seq):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        
        return self.input_dim  # If no Linear layer found

    def get_output_shape(self, input_shape: torch.Size):
        return input_shape[:-1] + (self.get_output_dim(),)
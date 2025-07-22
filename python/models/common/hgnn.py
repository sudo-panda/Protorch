import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GATv2Conv, GCNConv

"""
Heterogeneous Graph Neural Network (HGNN) model class.

Sample config:
[
    { "name": "SAGEConv", "out_channels": 128, "activation": "relu" },
    { "name": "GATConv",  "out_channels": 256, "activation": "tanh" },
    { "name": "GATv2Conv", "out_channels": 512, "activation": None },
    { "name": "GCNConv",   "out_channels": 256, "activation": "sigmoid" }
]
"""

class HGNN(nn.Module):
    hetero_conv_map = {
        "SAGEConv"  : SAGEConv,
        "GATConv"   : GATConv,
        "GATv2Conv" : GATv2Conv,
        "GCNConv"   : GCNConv,
    }
    
    activation_map = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        None: nn.Identity()
    }

    def __init__(self, config, input_dims, aggr="mean"):
        """
        - config: dict with [ {name, out_channels, activation}, ... ]
        - input_dims: dict {edge_type: initial_feature_dim}
        """
        super().__init__()
        self.edge_types = list(input_dims.keys())

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for layer_cfg in config:
            conv_name = layer_cfg["name"]
            out_ch = layer_cfg["out_channels"]
            act_name = layer_cfg.get("activation", "relu")

            conv_cls = self.hetero_conv_map[conv_name]
            # Build edge-specific convs with inferred input dims
            convs = {
                edge_type: conv_cls((-1, -1), out_ch)
                for edge_type in self.edge_types
            }

            self.layers.append(HeteroConv(convs, aggr=aggr))
            self.activations.append(self.activation_map.get(act_name, nn.Identity()))

    def forward(self, x_dict, edge_index_dict):
        for conv, act in zip(self.layers, self.activations):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: act(x) for nt, x in x_dict.items()}
        return x_dict

    def get_output_dim(self):
        for layer in reversed(self.layers):
            if isinstance(layer, HeteroConv):
                return int(list(layer.convs.values())[0].out_channels)

        raise ValueError("No output dim found")

    def get_output_shape(self, x_dict_shape_orig):
        x_dict_shape = {node: dim for node, dim in x_dict_shape_orig.items()}
        for layer in reversed(self.layers):
            if isinstance(layer, HeteroConv):
                for edge_type, conv in layer.convs.items():
                    dst = edge_type[2]
                    if dst in x_dict_shape:
                        x_dict_shape[dst] = torch.Size([x_dict_shape[dst][0], int(conv.out_channels)])

                break
        return x_dict_shape

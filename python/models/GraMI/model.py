from typing import Any
from torch import nn
import torch
from torch_geometric.data import HeteroData

from models.GraMI.encoder import GraMIEncoder
from models.GraMI.decoder import GraMIDecoder


class GraMIReparameterize(nn.Module):
    def __init__(self, ratio):
        super(GraMIReparameterize, self).__init__()
        self.ratio = ratio

    @staticmethod
    def reparameterize(z, ratio):
        z_mu, z_var = z
        eps = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5 * z_var) * eps * ratio

    def forward(self, z_A: tuple[torch.Tensor], z_V: dict[str, tuple[torch.Tensor]]):
        return \
            GraMIReparameterize.reparameterize(z_A, self.ratio), \
            { node: GraMIReparameterize.reparameterize(z_v, self.ratio) for node, z_v in z_V.items()}

class GraMIModel(nn.Module):
    def __init__(self, 
                 config: dict[str, dict], 
                 data_shapes: dict[str, dict],
                 extra_config: dict[str, Any] = {"ratio": 1.0}):
        super(GraMIModel, self).__init__()
        self.config = config

        self.is_variational = bool(len(self.config["node_encoder"]["variational"]) > 0)
        assert self.is_variational == bool(len(self.config["attribute_encoder"]["variational"]) > 0)

        self.encoder = GraMIEncoder(self.config, data_shapes)

        if self.is_variational:
            self.reparameterize = GraMIReparameterize(extra_config["ratio"])

        if "decoder" not in self.config:
            self.decoder_config = GraMIModel.invert_config(self.config)
            transforms_output_dim = self.encoder.transforms.get_output_dim()
            for node, layers in self.decoder_config["mlp"].items():
                layers.append({"Linear": [transforms_output_dim[node]]})
        else:
            self.decoder_config = self.config["decoder"]

        self.decoder = GraMIDecoder(self.decoder_config, data_shapes)

    @staticmethod
    def invert_config(config: dict[str, dict]) -> dict[str, dict]:
        inverted_config = {
            "hgnn": GraMIModel.invert_hgnn_config(config["node_encoder"]["layers"]),
            "mlp": {node_name: GraMIModel.invert_mlp_config(node_config) for node_name, node_config in config["init"].items()},
        }
        return inverted_config

    @staticmethod
    def invert_hgnn_config(config: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        config = [
            { "name": "SAGEConv", "out_channels": 1024, "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 512,  "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 256,  "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 128,  "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 64,   "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 32,   "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 16,   "activation": "relu" }
        ]

        will be inverted to

        inverted_config = [
            { "name": "SAGEConv", "out_channels": 16,   "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 32,   "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 64,   "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 128,  "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 256,  "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 512,  "activation": "relu" },
            { "name": "SAGEConv", "out_channels": 1024, "activation": "relu" }
        ]
        """
        inverted_config = list(reversed(config))
        return inverted_config

    @staticmethod
    def invert_mlp_config(config: list[dict[str, list]]) -> list[dict[str, list]]:
        """
        config = [
            { "Linear":     [128] },
            { "ReLU":       []    },
            { "LayerNorm":  [128] },
            { "Dropout":    [0.1] },
            { "Linear":     [256] },
            { "ReLU":       []    },
            { "LayerNorm":  [256] },
            { "Dropout":    [0.1] },
            { "Linear":     [512] },
            { "ReLU":       []    },
            { "LayerNorm":  [512] },
            { "Dropout":    [0.1] }
        ]

        will be inverted to

        inverted_config = [
            { "Linear":     [512] },
            { "ReLU":       []    },
            { "LayerNorm":  [512] },
            { "Dropout":    [0.1] },
            { "Linear":     [256] },
            { "ReLU":       []    },
            { "LayerNorm":  [256] },
            { "Dropout":    [0.1] },
            { "Linear":     [128] },
            { "ReLU":       []    },
            { "LayerNorm":  [128] },
            { "Dropout":    [0.1] }
        ]
        """
        inverted_config = []
        layer_set = []
        for layer in config:
            if "Linear" in layer:
                inverted_config = layer_set + inverted_config
                layer_set = []
            layer_set.append(layer)
        inverted_config = layer_set + inverted_config
        return inverted_config

    def forward(self, graph: HeteroData):
        x, x_tile, n_A, n_V = self.encoder(graph)
        
        if self.is_variational:
            z_A, z_V = self.reparameterize(n_A, n_V)
        else:
            z_A, z_V = n_A, n_V

        graph_ptr = {node: graph[node].ptr for node in graph.x_dict.keys()}
        edge_logits, x_tile_rec, x_rec = self.decoder(z_A, z_V, graph.edge_index_dict, graph_ptr)
        return x, x_tile, n_A, n_V, edge_logits, x_tile_rec, x_rec
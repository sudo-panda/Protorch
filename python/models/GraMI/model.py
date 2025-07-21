from torch import nn
import torch
from models.common import Transforms
from torch_geometric.data import HeteroData

from models.GraMI.encoder import GraMIEncoder
from models.GraMI.decoder import GraMIDecoder


class GraMIReparameterize(nn.Module):
    def __init__(self):
        super(GraMIReparameterize, self).__init__()

    @staticmethod
    def reparameterize(z):
        z_mu, z_var = z
        eps = torch.randn_like(z_mu)
        return z_mu + eps * torch.exp(0.5 * z_var)

    def forward(self, z_A: tuple[torch.Tensor], z_V: dict[str, tuple[torch.Tensor]]):
        return \
            GraMIReparameterize.reparameterize(z_A), \
            { node: GraMIReparameterize.reparameterize(z_v) for node, z_v in z_V.items()}

class GraMI(nn.Module):
    def __init__(self, sample_shape, config, device, batch_size):
        super(GraMI, self).__init__()
        self.config = config
        self.device = device
        self.batch_size = batch_size

        self.is_variational = bool(len(self.config["node_encoder"]["variational"]) > 0)
        assert self.is_variational == bool(len(self.config["attribute_encoder"]["variational"]) > 0)

        self.encoder = GraMIEncoder(sample_shape, self.config, self.device, self.batch_size)

        if self.is_variational:
            self.reparameterize = GraMIReparameterize()

        if "decoder" not in self.config:
            decoder_config = GraMI.invert_config(self.config)
        else:
            decoder_config = self.config["decoder"]

        self.decoder = GraMIDecoder(sample_shape, decoder_config, self.device, self.batch_size)

    @staticmethod
    def invert_config(config):
        inverted_config = {
            "hgnn": GraMI.invert_hgnn_config(config["node_encoder"]["layers"]),
            "mlp": {node_name: GraMI.invert_mpl_config(node_config) for node_name, node_config in config["init"].items()},
        }
        return inverted_config

    @staticmethod
    def invert_hgnn_config(config):
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
    def invert_mpl_config(config):
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
        x, x_tile, z_A, z_V = self.encoder(graph)
        
        if self.is_variational:
            z_A, z_V = self.reparameterize(z_A, z_V)
            
        edge_logits, x_tile_rec, x_rec = \
            self.decoder(z_A, z_V, graph.edge_index_dict,
                         {node: graph[node].ptr for node in graph.x_dict.keys()})
        return x, x_tile, z_A, z_V, edge_logits, x_tile_rec, x_rec
from torch import nn
import torch
from models.common import MLP
from models.GraMI.encoder import GraMIEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.data import HeteroData

class DevmapClassifier(nn.Module):
    def __init__(self, config, input_dim):
        super(DevmapClassifier, self).__init__()
        self.config = config

        self.linear = MLP(config, input_dim)

    def forward(self, x):
        return self.linear(x)

class DevmapE2EModel(nn.Module):
    def __init__(self, config, data_shapes):
        super(DevmapE2EModel, self).__init__()
        self.config = config

        self.node_order = list(data_shapes["x_dict"].keys())

        self.encoder = GraMIEncoder(config, data_shapes)
        self.pooled_dim = self.encoder.get_output_dim() * (len(self.node_order) + 1)
        self.classifier = DevmapClassifier(config["classifier"], self.pooled_dim)

    def forward(self, graph: HeteroData):
        _, _, z_A, z_V = self.encoder(graph)

        classifier_input = [ z_A.sum(dim=1) ]
        for node_type in self.node_order:
            classifier_input.append(global_add_pool(z_V[node_type], graph[node_type].batch))

        classifier_input = torch.cat(classifier_input, dim=1)

        logits = self.classifier(classifier_input)
        return logits
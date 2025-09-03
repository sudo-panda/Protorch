from typing import Any
from torch import nn
import torch
from models.GraMI.encoder import GraMIEncoder
from models.common import MLP, pool_enc_outs
from torch_geometric.data import HeteroData


class VecParamsClassifier(nn.Module):
    def __init__(self, config, data_shapes, extra_config: dict[str, Any] = {}):
        super(VecParamsClassifier, self).__init__()
        self.config = config
        self.node_order = sorted(list(data_shapes["z_V"].keys()))

        self.pooled_dim = data_shapes["z_A"][-1] * (len(self.node_order) + 1)
        self.mlp_input_dim = self.pooled_dim
        self.mlp = MLP(config, self.mlp_input_dim)

    def forward(self, z_A, z_V, batch, batch_size):
        """
        Returns:
            vf_logits: The logits for the vector width head.
            if_logits: The logits for the interleaving factor head.
        """
        outs = pool_enc_outs(z_A, z_V, batch, batch_size, self.node_order)

        mlp_inp = torch.cat(outs, dim=1)
        logits = self.mlp(mlp_inp)
        return logits


class VecParamsE2EModel(nn.Module):
    def __init__(self, config, data_shapes, extra_config: dict[str, Any] = {}):
        super(VecParamsE2EModel, self).__init__()
        self.config = config

        self.encoder = GraMIEncoder(config, data_shapes)

        output_shape = self.encoder.get_output_shape(data_shapes)

        data_shapes = {}
        if self.encoder.variational:
            data_shapes["z_A"] = output_shape["n_A"][0]
            data_shapes["z_V"] = {k: v[0] for k, v in output_shape["n_V"].items()}
        else:
            data_shapes["z_A"] = output_shape["n_A"]
            data_shapes["z_V"] = output_shape["n_V"]

        self.node_order = list(data_shapes["z_V"].keys())

        self.classifier = VecParamsClassifier(config["classifier"], data_shapes)

    def forward(self, graph: HeteroData):
        _, _, z_A, z_V = self.encoder(graph)

        batch = {k: graph[k].batch for k in self.node_order}
        batch_size = len(graph[list(graph.x_dict.keys())[0]].ptr) - 1

        logits = self.classifier.forward(z_A, z_V, batch, batch_size)
        return logits

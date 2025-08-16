from torch import nn
import torch
from models.common import MLP
from models.GraMI.encoder import GraMIEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.data import HeteroData

class DevmapClassifier(nn.Module):
    def __init__(self, config, enc_input_dim, enc_node_types):
        super(DevmapClassifier, self).__init__()
        self.config = config
        self.node_order = sorted(enc_node_types)

        self.pooled_dim = enc_input_dim * (len(self.node_order) + 1)
        self.mlp_input_dim = self.pooled_dim + 6 # +6 for comp, mem, localmem, coalesced, transfer, wgsize
        self.mlp = MLP(config, self.mlp_input_dim)

    def forward(self, z_A, z_V,
                comp, mem, localmem, 
                coalesced, transfer, wgsize,  
                batch):
        mlp_inp = [ z_A.sum(dim=1) ]
        for node_type in self.node_order:
            mlp_inp.append(global_add_pool(z_V[node_type], batch[node_type]))

        mlp_inp.append(comp.unsqueeze(-1))
        mlp_inp.append(mem.unsqueeze(-1))
        mlp_inp.append(localmem.unsqueeze(-1))
        mlp_inp.append(coalesced.unsqueeze(-1))
        mlp_inp.append(transfer.unsqueeze(-1))
        mlp_inp.append(wgsize.unsqueeze(-1))

        mlp_inp = torch.cat(mlp_inp, dim=1)
        return self.mlp(mlp_inp)

class DevmapE2EModel(nn.Module):
    def __init__(self, config, data_shapes):
        super(DevmapE2EModel, self).__init__()
        self.config = config

        self.node_order = list(data_shapes["x_dict"].keys())

        self.encoder = GraMIEncoder(config, data_shapes)

        self.classifier = DevmapClassifier(config["classifier"], self.encoder.get_output_dim(), self.node_order)

    def forward(self, graph: HeteroData):
        _, _, z_A, z_V = self.encoder(graph)

        batch = {k: graph[k].batch for k in self.node_order}
        
        logits = self.classifier(
            z_A, z_V, 
            graph.comp, graph.mem, graph.localmem, 
            graph.coalesced, graph.transfer, graph.wgsize, 
            batch)
        return logits
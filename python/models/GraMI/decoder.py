from platform import node
from torch import nn
import torch
from torch_geometric.data import HeteroData
from models.common import MLP, HGNN



class GraMINodeDecoder(nn.Module):
    def __init__(self, edge_index_shape):
        super(GraMINodeDecoder, self).__init__()
        self.edge_index_shape = edge_index_shape

    def forward(self, z_V: dict[str, torch.Tensor]):
        edge_logits = {}
        for edge_type in self.edge_index_shape:
            edge_logits[edge_type] = torch.sigmoid(torch.matmul(z_V[edge_type[2]], z_V[edge_type[0]].T))
        return edge_logits

class GraMIAttributeDecoder(nn.Module):
    def __init__(self, attr_dim, config, device, batch_size):
        super(GraMIAttributeDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        self.batch_size = batch_size

        self.hgnn = HGNN(config["hgnn"], attr_dim["edge_index_dict"]).to(device=self.device)
        self.mlp = {
            node_type: MLP(node_config, self.hgnn.get_output_dim()).to(device=self.device)
            for node_type, node_config in config["mlp"].items()
        }

    @staticmethod
    def unbatch_graphs(batched, ptrs, batch_size):
        unbatched = [{} for _ in range(batch_size)]
        for node_type, graph in batched.items():
            for i in range(batch_size):
                unbatched[i][node_type] = graph[ptrs[node_type][i]:ptrs[node_type][i + 1]]
        
        return unbatched
    

    @staticmethod
    def rebatch_graphs(unbatched, batch_size):
        rebatch = {}
        for i in range(batch_size):
            for node_type, tensor in unbatched[i].items():
                if node_type not in rebatch:
                    rebatch[node_type] = []
                rebatch[node_type].append(tensor)

        for node_type, tensors in rebatch.items():
            rebatch[node_type] = torch.cat(tensors, dim=0)
        
        return rebatch

    def forward(self, z_A, z_V, edge_index_dict, ptrs):
        z_V_unbatched = GraMIAttributeDecoder.unbatch_graphs(z_V, ptrs, self.batch_size)

        z_rec = [{} for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for node_type, z_Vi in z_V_unbatched[i].items():
                z_rec[i][node_type] = torch.tanh(torch.matmul(z_Vi, z_A[i].T))

        z_rec = GraMIAttributeDecoder.rebatch_graphs(z_rec, self.batch_size)

        assert {node_type: z_V[node_type].shape[0] for node_type in z_V} == \
               {node_type: z_rec[node_type].shape[0] for node_type in z_rec}, \
               "Number of nodes in rebatched z_rec do not match number of nodes in original z_V"

        x_tile_rec = self.hgnn(z_rec, edge_index_dict)

        x_rec = {node_type: self.mlp[node_type](x) for node_type, x in x_tile_rec.items()}
        return x_tile_rec, x_rec

class GraMIDecoder(nn.Module):
    def __init__(self, sample_shape, config, device, batch_size):
        super(GraMIDecoder, self).__init__()
        self.config = config
        self.device = device
        self.batch_size = batch_size

        self.node_decoder = GraMINodeDecoder(sample_shape["edge_index_dict"])

        self.attribute_decoder = GraMIAttributeDecoder(sample_shape, config, device, batch_size)

    def forward(self, 
                z_A: torch.Tensor, 
                z_V: dict[str, torch.Tensor], 
                edge_index_dict: dict[str, torch.Tensor], 
                ptrs: dict[str, torch.Tensor]):
        edge_logits = self.node_decoder(z_V)
        x_tile_rec, x_rec = self.attribute_decoder(z_A, z_V, edge_index_dict, ptrs)
        return edge_logits, x_tile_rec, x_rec
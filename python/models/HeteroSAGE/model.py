import torch
from torch.nn import ModuleDict, Module
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GCNConv
from torch_geometric.nn.pool import global_mean_pool


class HeteroSAGE(Module):
    def __init__(self, hidden_channels, out_channels):
        super(HeteroSAGE, self).__init__()

        self.conv1 = HeteroConv({
            ('value', 'w_type', 'typ'): SAGEConv((-1, -1), hidden_channels),
            ('instruction', 'dataflow', 'value'): SAGEConv((-1, -1), hidden_channels),
            ('value', 'dataflow', 'instruction'): SAGEConv((-1, -1), hidden_channels),
            ('value', 'w_attribute', 'attribute'): SAGEConv((-1, -1), hidden_channels),
            ('typ', 'w_size', 'size'): SAGEConv((-1, -1), hidden_channels),
            # ('module', 'symbol', 'value'): GCNConv(-1, hidden_channels, add_self_loops=False),
            ('instruction', 'cfg', 'instruction'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.node_lin = ModuleDict({
            node_type: Linear(hidden_channels, hidden_channels)
            for node_type in ['value', 'typ', 'size', 'module', 'attribute', 'instruction']
        })

        self.graph_lin = Linear(hidden_channels * 5, out_channels)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv1.forward(x, edge_index)

        x_relu = {key: x.relu() for key, x in x_conv.items()}

        node_embeddings = torch.cat(
            [global_mean_pool(self.node_lin[node_type](node), batch[node_type])
             for node_type, node in x_relu.items()], dim=-1)

        graph_embedding = self.graph_lin(node_embeddings)

        return graph_embedding

def get_model() -> Module:
    return HeteroSAGE(hidden_channels=64, out_channels=64)

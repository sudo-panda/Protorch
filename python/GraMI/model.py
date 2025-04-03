import torch
from torch.nn import ModuleDict, Module, Linear, Tanh, BatchNorm1d, Sequential
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GATv2Conv, InnerProductDecoder
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import HeteroData

class MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            output_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(Linear(input_dim, output_dim))
            layers.append(Tanh())
            layers.append(BatchNorm1d(output_dim))
        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class HGNN(Module):
    def __init__(self, data: HeteroData, hidden_channels, out_channels, num_layers=1):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: GATv2Conv((-1, -1), hidden_channels if i != num_layers - 1 else out_channels, add_self_loops=False) for edge_type in data.edge_index_dict.keys()
            }, aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

class GraMIInitializer(Module): 
    def __init__(self, data, out_dim):
        super(GraMIInitializer, self).__init__()

        self.MLP1 = ModuleDict({
            node_type: MLP(data.size(1), (data.size(1) + 1) // 2, out_dim)
            for node_type, data in data.x_dict.items()
        })
    
    def forward(self, x_dict):
        return {node_type: self.MLP1[node_type](x) for node_type, x in x_dict.items()}
        
class GraMIEncoder(Module):
    def __init__(self, hetero_data, in_dim, out_dim):
        super(GraMIEncoder, self).__init__()
        
        self.hgnn = HGNN(hetero_data, in_dim, out_dim, num_layers=1)

        self.node_mu = Linear(out_dim, out_dim)
        self.node_logvar = Linear(out_dim, out_dim)

        self.mlp = MLP(in_dim, in_dim, out_dim, num_layers=3)

        self.attr_mu = Linear(out_dim, out_dim)
        self.attr_logvar = Linear(out_dim, out_dim)
    
    def forward(self, x_dict, edge_index):
        X_eps = {k: x + torch.normal(0, 1, size=x.shape) for k, x in x_dict.items()}
        h_V = self.hgnn(X_eps, edge_index)        
        V = {node_type: (self.node_mu(h), self.node_logvar(h)) for node_type, h in h_V.items()}

        X_hat = torch.cat([x_dict[node_type] for node_type in x_dict.keys()], dim=0)
        X_hat_eps = X_hat + torch.normal(0, 1, size=X_hat.shape)
        h_A = self.mlp(X_hat_eps)
        A = (self.attr_mu(h_A), self.attr_logvar(h_A))

        return V, A
        

class GraMIDecoder(Module):
    def __init__(self, data: HeteroData, hidden_dim):
        super(GraMIDecoder, self).__init__()

        self.hgnn = HGNN(data, hidden_dim // 2, hidden_dim, num_layers=1)

        self.MLP1 = ModuleDict({
            node_type: MLP(hidden_dim, (data.size(1) + 1) // 2, data.size(1), num_layers=2)
            for node_type, data in data.x_dict.items()
        })
    
    def forward(self, Z_V, Z_A, edge_index):
        edge_logits = {}
        for edge_type in edge_index.keys():
            edge_logits[edge_type] = torch.sigmoid(torch.matmul(Z_V[edge_type[2]], Z_V[edge_type[0]].T))

        Z_prime = {}
        for node_type, Z_Vi in Z_V.items():
            Z_prime[node_type] = torch.tanh(torch.matmul(Z_Vi, Z_A.T))

        X_hat_prime = self.hgnn(Z_prime, edge_index)

        X_prime = {node_type: self.MLP1[node_type](x) for node_type, x in X_hat_prime.items()}

        return X_hat_prime, edge_logits, X_prime

class GraMIModel(Module):
    def __init__(self, data: HeteroData, hidden_dim, encode_dim):
        super(GraMIModel, self).__init__()
        self.initializer = GraMIInitializer(data, hidden_dim)
        self.encoder = GraMIEncoder(data, hidden_dim, encode_dim)
        self.decoder = GraMIDecoder(data, hidden_dim)
    
    def reparameterize(self, V, A):
        Z_V = {k: v[0] + torch.normal(0, 1, size=v[1].shape) * torch.exp(0.5 * v[1]) for k, v in V.items()}
        Z_A = A[0] + torch.normal(0, 1, size=A[1].shape) * torch.exp(0.5 * A[1])

        return Z_V, Z_A
    
    def forward(self, X, edge_index_dict):
        X_hat = self.initializer(X)
        V, A = self.encoder(X_hat, edge_index_dict)
        
        Z_V, Z_A = self.reparameterize(V, A)
        
        X_hat_prime, edge_logits, X_prime = self.decoder(Z_V, Z_A, edge_index_dict)
        
        return X_hat, V, A, X_hat_prime, edge_logits, X_prime
    


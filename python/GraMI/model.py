import torch
from torch.nn import ModuleDict, Module, Linear, Tanh, Sequential, AdaptiveAvgPool1d, LayerNorm, Embedding, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GATv2Conv, InnerProductDecoder
from utils.digit_embeddings import get_digit_emb_of_number, feat_count
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import HeteroData

class MLP(Module):
    def __init__(self, layer_channels: list[int]):
        super().__init__()

        assert len(layer_channels) > 1, "MLP must have at least 2 channels (input and output)"

        layers = []
        num_layers = len(layer_channels) - 1

        for i in range(num_layers):
            input_dim = layer_channels[i]
            output_dim = layer_channels[i + 1]
            layers.append(LayerNorm(input_dim))
            layers.append(Linear(input_dim, output_dim))
            layers.append(Tanh())
        
        self.seq = Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class HGNN(Module):
    def __init__(self, edges, layer_out_channels: list[int]):
        super().__init__()

        num_layers = len(layer_out_channels)
        assert num_layers > 0, "HGNN must have at least 1 layer"
        
        self.convs = ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv = HeteroConv({
                    edge_type: SAGEConv((-1, -1), layer_out_channels[i]) for edge_type in edges
                }, aggr='mean')   
            else:
                conv = HeteroConv({
                    edge_type: GATv2Conv((-1, -1), layer_out_channels[i], add_self_loops=False) for edge_type in edges
                }, aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

class GraMIInitializer(Module): 
    def __init__(self, data_dims, hidden_dim_per_node: dict[str, list[int]], init_to_dim: int, digit_embed_size = 10):
        super(GraMIInitializer, self).__init__()
        
        self.embeds = Embedding(feat_count, digit_embed_size)

        self.MLP1 = ModuleDict({
            node_type: MLP([
                data_dims[node_type], 
                *(hidden_dim_per_node[node_type] if node_type in hidden_dim_per_node else []), 
                init_to_dim
            ]) for node_type in data_dims.keys()
        })
    
    def forward(self, x_dict, text_attrs):
        number_embeds = []
        for number_list in text_attrs["number"]:
            if not isinstance(number_list, list):
                number_list = [number_list]
            for number in number_list:
                number_embeds.append(get_digit_emb_of_number(number, self.embeds).unsqueeze(0).to(x_dict["number"].device))
        number_embeds = torch.cat(number_embeds)

        x_dict["number"] = number_embeds

        return x_dict, {node_type: self.MLP1[node_type](x) for node_type, x in x_dict.items()}
        
class GraMIEncoder(Module):
    def __init__(self, edges, hgnn_hidden_channels, mlp_hidden_channels, encode_dim):
        super(GraMIEncoder, self).__init__()
        
        self.hgnn_enc = HGNN(edges, [*hgnn_hidden_channels, encode_dim])

        self.node_mu = Linear(encode_dim, encode_dim)
        self.node_logvar = Linear(encode_dim, encode_dim)

        self.mlp_2 = Sequential(AdaptiveAvgPool1d(mlp_hidden_channels[0]), MLP([*mlp_hidden_channels, encode_dim]))

        self.attr_mu = Linear(encode_dim, encode_dim)
        self.attr_logvar = Linear(encode_dim, encode_dim)
    
    def forward(self, x_dict, edge_index):
        X_eps = {k: x + torch.normal(0, 1, size=x.shape, device=x.device) for k, x in x_dict.items()}
        h_V = self.hgnn_enc(X_eps, edge_index)
        V = {node_type: (self.node_mu(h), self.node_logvar(h)) for node_type, h in h_V.items()}

        X_hat = torch.cat([x_dict[node_type] for node_type in x_dict.keys()], dim=0)
        X_hat_eps = X_hat + torch.normal(0, 1, size=X_hat.shape, device=X_hat.device)
        h_A = self.mlp_2(X_hat_eps.T)
        A = (self.attr_mu(h_A), self.attr_logvar(h_A))

        return V, A
        

class GraMIDecoder(Module):
    def __init__(self, edges, data_dims, init_dim, hidden_dim: dict[str, list[int]]):
        super(GraMIDecoder, self).__init__()

        # HGNN to decode the Edge reconstruction and Attribute reconstruction into the 
        # Node features that we get after GraMIInitializer
        # Encoded Dim -> Init Dim
        self.hgnn_dec = HGNN(edges, [init_dim])

        # MLP to decode the refonstructed Init Node features into the original Node features
        # Init Dim -> Original Dim
        self.MLP3 = ModuleDict({
            node_type: MLP([init_dim, *(hidden_dim[node_type] if node_type in hidden_dim else []), dim])
            for node_type, dim in data_dims.items()
        })
    
    def forward(self, Z_V, Z_A, edge_index):
        edge_logits = {}
        for edge_type in edge_index.keys():
            edge_logits[edge_type] = torch.sigmoid(torch.matmul(Z_V[edge_type[2]], Z_V[edge_type[0]].T))

        Z_prime = {}
        for node_type, Z_Vi in Z_V.items():
            Z_prime[node_type] = torch.tanh(torch.matmul(Z_Vi, Z_A.T))

        X_hat_prime = self.hgnn_dec(Z_prime, edge_index)

        X_prime = {node_type: self.MLP3[node_type](x) for node_type, x in X_hat_prime.items()}

        return X_hat_prime, edge_logits, X_prime

class GraMIModel(Module):
    def __init__(self, data: HeteroData, args):
        super(GraMIModel, self).__init__()

        digit_embed_size = args["digit_embed_size"]

        data_dims = {node_type: data.size(1) for node_type, data in data.x_dict.items()}
        data_dims["number"] = digit_embed_size
        
        edges = [edge_type for edge_type in data.edge_index_dict.keys()]

        self.initializer = GraMIInitializer(data_dims, args["init"]["hidden_dims"], args["init"]["final_dim"], digit_embed_size)
        self.encoder = GraMIEncoder(edges, args["encoder"]["hgnn_dims"], args["encoder"]["mlp_dims"], args["encoder"]["final_dim"])
        self.decoder = GraMIDecoder(edges, data_dims, args["init"]["final_dim"], {node_type: args["init"]["hidden_dims"][node_type][::-1] for node_type in data_dims.keys()})
    
    def reparameterize(self, V, A):
        Z_V = {k: v[0] + torch.normal(0, 1, size=v[1].shape, device=v[1].device) * torch.exp(0.5 * v[1]) for k, v in V.items()}
        Z_A = A[0] + torch.normal(0, 1, size=A[1].shape, device=A[1].device) * torch.exp(0.5 * A[1])

        return Z_V, Z_A
    
    def forward(self, x_dict, edge_index_dict, text_attrs):
        X, X_hat = self.initializer(x_dict, text_attrs)
        V, A = self.encoder(X_hat, edge_index_dict)
        
        Z_V, Z_A = self.reparameterize(V, A)
        
        X_hat_prime, edge_logits, X_prime = self.decoder(Z_V, Z_A, edge_index_dict)
        
        return X, X_hat, V, A, X_hat_prime, edge_logits, X_prime
    


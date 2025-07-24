from torch import nn
import torch
from models.common import Transforms
from torch_geometric.data import HeteroData
from models.common import MLP, HGNN

class GraMIInit(nn.Module):
    def __init__(self, config, x_dict_shape, device):
        super(GraMIInit, self).__init__()
        self.config = config
        self.device = device

        self.layer_dict = nn.ModuleDict({
            node_name: MLP(layer_config, x_dict_shape[node_name][-1]).to(device)
            for node_name, layer_config in config.items()
        })

    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        for node_name, layer in self.layer_dict.items():
            out_dict[node_name] = layer(x_dict[node_name])
        return out_dict

    def get_output_shape(self, x_dict_shape: dict[str, torch.Size]):
        return {node_name: layer.get_output_shape(x_dict_shape[node_name]) # type: ignore 
                for node_name, layer in self.layer_dict.items()}

class GraMIAttributeEncoder(nn.Module):
    def __init__(self, pool_dim, config, device, batch_size, variational=[], stochastic=False):
        super(GraMIAttributeEncoder, self).__init__()
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.stochastic = stochastic
        self.variational = bool(len(variational) > 0)

        self.pool_dim = pool_dim

        self.pool = nn.AdaptiveAvgPool1d(self.pool_dim)

        self.mlp = MLP(self.config, self.pool_dim).to(device)

        if self.stochastic:
            self.mlp_eps = MLP(self.config, self.pool_dim).to(device)

        ae_dim = self.mlp.get_output_dim()

        if self.variational:
            self.mlp_mean = MLP(variational, ae_dim).to(device)
            self.mlp_var  = MLP(variational, ae_dim).to(device)

    def forward(self, X_T: list[torch.Tensor]):
        # TODO: Convert X_T from [(512, 661), (512, 2275), (512, 1086), (512, 1332)]
        #                     to [(512, dim), (512,  dim), (512,  dim), (512,  dim)]

        # Option 1: pad by zeros to get dim

        # Option 2: Adaptive Avg Pooling
        X_T_pooled = torch.stack([self.pool(x_T) for x_T in X_T], dim=0)

        n_A = self.mlp(X_T_pooled)

        if self.stochastic:
            eps = torch.randn_like(X_T_pooled)

            n_eps = self.mlp_eps(eps)

            n_A = n_A + n_eps

        if self.variational:
            n_A_mean = self.mlp_mean(n_A)
            n_A_var  = self.mlp_var(n_A)
            n_A = (n_A_mean, n_A_var)


        X_T_shape = [x_T.shape for x_T in X_T]
        assert all([z.shape == self.get_output_shape(X_T_shape) for z in n_A]) if isinstance(n_A, tuple) else n_A.shape == self.get_output_shape(X_T_shape)
        return n_A # (B, F, N)

    def get_output_dim(self):
        if self.variational:
            return self.mlp_mean.get_output_dim()
        
        return self.mlp.get_output_dim()

    def get_output_shape(self, X_T_shape):
        batch_size = len(X_T_shape)

        X_T_pooled_shape = torch.Size([batch_size, X_T_shape[0][0], self.pool_dim * batch_size])

        if self.variational:
            return self.mlp_mean.get_output_shape(X_T_pooled_shape)

        return self.mlp.get_output_shape(X_T_pooled_shape)

class GraMINodeEncoder(nn.Module):
    def __init__(self, config, edge_index_dict_shape, device, variational=[], stochastic=False):
        super(GraMINodeEncoder, self).__init__()
        self.config = config
        self.device = device
        self.stochastic = stochastic
        self.variational = (len(variational) > 0)

        self.hgnn = HGNN(config, edge_index_dict_shape).to(device)

        if variational:
            self.mlp_mean = MLP(variational, self.hgnn.get_output_dim()).to(device)
            self.mlp_var  = MLP(variational, self.hgnn.get_output_dim()).to(device)

        if stochastic:
            self.hgnn_eps = HGNN(config, edge_index_dict_shape).to(device)

    def forward(self, graph):
        n_V = self.hgnn(graph.x_dict, graph.edge_index_dict)

        if self.stochastic:
            eps = {node_name: torch.randn_like(x) for node_name, x in graph.x_dict.items()}

            hidden_eps = self.hgnn_eps(eps, graph.edge_index_dict)

            n_V = {node_name: (n_V[node_name] + hidden_eps[node_name]) for node_name in n_V}

        if self.variational:
            n_V = {node_name: (self.mlp_mean(z), self.mlp_var(z)) for node_name, z in n_V.items()}

        output_data_shape = {node: ((z[0].shape, z[1].shape) if isinstance(z, tuple) else z.shape) for node, z in n_V.items()}
        output_shape = self.get_output_shape({node: x.shape for node, x in graph.x_dict.items()})
        assert output_data_shape == output_shape, f"Output data shape mismatch: {output_data_shape} != {output_shape}"
        return n_V

    def get_output_dim(self):
        if self.variational:
            return self.mlp_mean.get_output_dim()

        return self.hgnn.get_output_dim()

    def get_output_shape(self, x_dict_shape):
        if self.variational:
            return {node: (self.mlp_mean.get_output_shape(x), self.mlp_var.get_output_shape(x)) for node, x in x_dict_shape.items()}

        return self.hgnn.get_output_shape(x_dict_shape)

class GraMIEncoder(nn.Module):
    def __init__(self, config, data_shapes, device, batch_size):
        super(GraMIEncoder, self).__init__()
        self.config = config
        self.device = device
        self.batch_size = batch_size

        transforms   = self.config["transforms"]
        attr_enc_cfg = self.config["attribute_encoder"]
        node_enc_cfg = self.config["node_encoder"]

        attr_enc_cfg_var = attr_enc_cfg.get("variational", [])
        node_enc_cfg_var = node_enc_cfg.get("variational", [])

        self.variational = bool(len(attr_enc_cfg_var) > 0)
        assert self.variational == bool(len(node_enc_cfg_var) > 0)

        self.node_order = list(data_shapes["x_dict"].keys())

        self.transforms = Transforms(transforms, data_shapes["x_dict"])
        data_shapes["x_dict"] = self.transforms.get_output_shape(data_shapes["x_dict"])

        self.init_layers = GraMIInit(self.config["init"], data_shapes["x_dict"], device)
        data_shapes["x_dict"] = self.init_layers.get_output_shape(data_shapes["x_dict"])

        self.attribute_encoder = GraMIAttributeEncoder(attr_enc_cfg["dim"], 
                                                       attr_enc_cfg["layers"], 
                                                       self.device, 
                                                       self.batch_size,
                                                       variational=attr_enc_cfg_var,
                                                       stochastic=attr_enc_cfg["stochastic"])
        X_T_shape = GraMIEncoder.get_X_t_shape(data_shapes, self.node_order)
        attr_enc_shape = self.attribute_encoder.get_output_shape(X_T_shape)


        self.node_encoder = GraMINodeEncoder(node_enc_cfg["layers"],
                                             data_shapes["edge_index_dict"], 
                                             device,
                                             variational=node_enc_cfg_var, 
                                             stochastic=node_enc_cfg["stochastic"])
        node_enc_shape = data_shapes.copy()
        node_enc_shape["x_dict"] = self.node_encoder.get_output_shape(node_enc_shape["x_dict"])

        assert self.attribute_encoder.get_output_dim() == self.node_encoder.get_output_dim(), "The output dimension of the Node encoder and Attribute encoder must match!"

    @staticmethod
    def get_X_t(graph, node_order):
        X_t = []
        for node_name in node_order:
            ptr = graph[node_name].ptr
            x = graph.x_dict[node_name]
            if len(X_t) == 0:
                for i in range(len(ptr) - 1):
                    X_t.append([])

            for i in range(len(ptr) - 1):
                X_t[i].append(x[ptr[i]:ptr[i + 1]])

        for i in range(len(X_t)):
            X_t[i] = torch.cat(X_t[i], dim=0).T

        return X_t
    
    @staticmethod
    def get_X_t_shape(data_shapes, node_order):
        X_t = []
        for node_name in node_order:
            ptr = data_shapes["ptr"][node_name]
            x = data_shapes["x_dict"][node_name]
            if len(X_t) == 0:
                for i in range(len(ptr) - 1):
                    X_t.append([x[-1], 0])

            for i in range(len(ptr) - 1):
                X_t[i][-1] += ptr[i + 1] - ptr[i]

        for i in range(len(X_t)):
            X_t[i] = torch.Size(X_t[i])

        return X_t

    def forward(self, graph: HeteroData):
        graph.x_dict = self.transforms(graph.x_dict, graph.text)
        x = {k: v.clone() for k, v in graph.x_dict.items()}

        graph.x_dict = self.init_layers(graph.x_dict, graph.edge_index_dict)
        x_tile  = {k: v.clone() for k, v in graph.x_dict.items()}

        X_t = GraMIEncoder.get_X_t(graph, self.node_order)
        n_A = self.attribute_encoder(X_t)

        n_V = self.node_encoder(graph)

        return x, x_tile, n_A, n_V

    def get_output_dim(self):
        return self.attribute_encoder.get_output_dim()

    def get_output_shape(self, data_shapes):
        x_shape = self.transforms.get_output_shape(data_shapes)
        x_tile_shape = self.init_layers.get_output_shape(data_shapes)
        X_t_shape = GraMIEncoder.get_X_t_shape(data_shapes, self.node_order)
        n_A_shape = self.attribute_encoder.get_output_shape(X_t_shape)
        n_V_shape = self.node_encoder.get_output_shape(x_shape)
        return {
            "x": x_shape,
            "x_tile": x_tile_shape,
            "n_A": n_A_shape,
            "n_V": n_V_shape
        }

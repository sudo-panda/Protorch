import datetime
from pathlib import Path
import argparse
import socket
import torch
import yaml
from torch_geometric.data import HeteroData

script_path = Path(__file__)
module_path = script_path.parent.parent.resolve()
config_path = module_path / "config.yaml"
models_path = module_path / "models"
config_dict = None


def get_data_shape(data: HeteroData):
    get_x_dict_shape = lambda x_dict : {key: value.shape for key, value in x_dict.items()}
    get_edge_index_shape = lambda edge_index_dict : {key: value.shape for key, value in edge_index_dict.items()}

    x_dict_shape = get_x_dict_shape(data.x_dict)
    edge_index_shape = get_edge_index_shape(data.edge_index_dict)
    ptr = {node: data[node].ptr for node in data.x_dict.keys()}
    
    return {"x_dict": x_dict_shape, "edge_index_dict": edge_index_shape, "ptr": ptr}

def get_log_dir_name(model_name):
    timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    log_dir_name = f"{timestamp}_{hostname}_{model_name}"
    return log_dir_name

def get_config():
    global config_dict

    if config_dict is None:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

    return config_dict

def get_adj_mat_from_edge_index(x_dict, edge_index_dict):
    adj_mat = {}
    for edge_typ, index in edge_index_dict.items():
        adj_mat[edge_typ] = torch.zeros(x_dict[edge_typ[2]].size(0), x_dict[edge_typ[0]].size(0), device=x_dict[edge_typ[0]].device)
        adj_mat[edge_typ][index[1], index[0]] = 1

    return adj_mat


config = get_config()

device = config["device"]
epochs =  config["train"]["epochs"]
train_from_checkpoint = config["train"]["train_from_checkpoint"]
lr = config["train"]["learning_rate"]
decay = config["train"]["weight_decay"]
batch_size = config["train"]["batch_size"]

if "world_size" in config["train"]:
    world_size = config["train"]["world_size"]

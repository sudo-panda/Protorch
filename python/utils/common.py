from pathlib import Path
import importlib
import torch
import yaml
import sys

script_path = Path(__file__)
module_path = script_path.parent.parent.resolve()
config_path = module_path / "config.yaml"
models_path = module_path / "models"
config_dict = None


def get_config():
    global config_dict

    print(config_path)

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
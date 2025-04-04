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

def get_model():
    if models_path not in sys.path:
        sys.path.append(str(models_path.absolute()))
    model = get_config()["model"]
    module = importlib.import_module(f"{model}.model")
    return module.get_model()

def get_adj_mat_from_edge_index(x_dict, edge_index_dict):
    adj_mat = {}
    for edge_typ, index in edge_index_dict.items():
        adj_mat[edge_typ] = torch.zeros(x_dict[edge_typ[2]].size(0), x_dict[edge_typ[0]].size(0), device=x_dict[edge_typ[0]].device)
        adj_mat[edge_typ][index[1], index[0]] = 1

    return adj_mat
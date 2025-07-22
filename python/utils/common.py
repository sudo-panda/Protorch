import datetime
import glob
import re
import socket

import torch
from torch_geometric.data import HeteroData


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

def get_adj_mat_from_edge_index(x_dict, edge_index_dict):
    adj_mat = {}
    for edge_typ, index in edge_index_dict.items():
        adj_mat[edge_typ] = torch.zeros(x_dict[edge_typ[2]].size(0), x_dict[edge_typ[0]].size(0), device=x_dict[edge_typ[0]].device)
        adj_mat[edge_typ][index[1], index[0]] = 1

    return adj_mat

def find_latest_file(pattern="file_*.pt"):
    files = glob.glob(pattern)
    versioned = []
    pattern = pattern.replace("*", "(\d+)")
    for f in files:
        m = re.match(pattern, f)
        if m:
            versioned.append((int(m.group(1)), f))
    if not versioned:
        return 0, None
    # pick the tuple with the largest version
    max_v = max(versioned, key= lambda t: t[0])
    return max_v

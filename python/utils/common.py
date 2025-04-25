import datetime
from pathlib import Path
import argparse
import socket
import torch
import yaml
import sys

script_path = Path(__file__)
module_path = script_path.parent.parent.resolve()
config_path = module_path / "config.yaml"
models_path = module_path / "models"
config_dict = None

def parse_args():
    parser = argparse.ArgumentParser(description="Train the GraMI model")
    parser.add_argument("--d_embed", type=int, default=10, help="Size of the digit embedding")
    parser.add_argument("--init_instr", type=str, default="", help="Hidden dims for initializer instruction")
    parser.add_argument("--init_val", type=str, default="", help="Hidden dims for initializer value")
    parser.add_argument("--init_num", type=str, default="", help="Hidden dims for initializer number")
    parser.add_argument("--init_typ", type=str, default="", help="Hidden dims for initializer type")
    parser.add_argument("--init_attr", type=str, default="", help="Hidden dims for initializer attribute")
    parser.add_argument("--init_size", type=str, default="", help="Hidden dims for initializer size")
    parser.add_argument("--init_fdim", type=int, default=16, help="Final dim for initializer")
    parser.add_argument("--enc_hgnn", type=str, default="", help="Hidden dims for encoder HGNN")
    parser.add_argument("--enc_mlp", type=str, default="64,16", help="Hidden dims for encoder MLP")
    parser.add_argument("--enc_fdim", type=int, default=8, help="Final dim for encoder")
    parser.add_argument("--mname", type=str, default="latest", help="Model name")
    arguments = parser.parse_args(sys.argv[1:])

    # Convert string arguments to appropriate types
    args = {
        "digit_embed_size": arguments.d_embed,
        "init": {
            "hidden_dims": {
                "instruction": list(map(int, arguments.init_instr.split(","))) if arguments.init_instr else [],
                "value": list(map(int, arguments.init_val.split(","))) if arguments.init_val else [],
                "number": list(map(int, arguments.init_num.split(","))) if arguments.init_num else [],
                "typ": list(map(int, arguments.init_typ.split(","))) if arguments.init_typ else [],
                "attribute": list(map(int, arguments.init_attr.split(","))) if arguments.init_attr else [],
                "size": list(map(int, arguments.init_size.split(","))) if arguments.init_size else [],
            },
            "final_dim": arguments.init_fdim,
        },
        "encoder": {
            "hgnn_dims": list(map(int, arguments.enc_hgnn.split(","))) if arguments.enc_hgnn else [],
            "mlp_dims": list(map(int, arguments.enc_mlp.split(","))) if arguments.enc_mlp else [],
            "final_dim": arguments.enc_fdim,
        },
        "model_name": arguments.mname,
    }

    # Sample arguments
    # args = {
    #     "digit_embed_size": 10,
    #     "init": {
    #         "hidden_dims": {
    #             "instruction": [64, 32],
    #             "value": [64, 32],
    #             "number": [],
    #             "typ": [32],
    #             "attribute": [],
    #             "size": [],
    #         },
    #         "final_dim": 16,
    #     },
    #     "encoder": {
    #         "hgnn_dims": [16, 16],
    #         "mlp_dims": [64, 16],
    #         "final_dim": 8,
    #     },
    # }

    print("Parsed arguments:", args)
    return args

def get_log_dir_name(model_name):
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    log_dir_name = f"{timestamp}_{hostname}_{model_name}"
    return log_dir_name

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
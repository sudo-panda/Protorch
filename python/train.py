import torch
import json

from utils.paths import get_data_paths
from utils.config import TrainConfig
from utils.train import (
    parse_and_run,
    setup_training,
    training_loop,
)

from utils.dataset import GraphDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from models.GraMI import GraMIModel, GraMI_loss, edge_and_r2_acc, SeeR_GraMI_loss

from utils.common import get_adj_mat_from_edge_index

def load_graph_data(dataset, device, batch_size, cfg):
    ################ Load data ################
    _, data_path, csv_file = get_data_paths(dataset)

    with open(csv_file) as f:
        df = pd.read_csv(f)

    df["file_path"] = df["pt_file"].apply(lambda x: str(data_path / x))
    file_list = df["file_path"].tolist()

    # train_files, temp_files = train_test_split(file_list,  test_size=0.4, random_state=cfg.seed)
    # val_files,   test_files = train_test_split(temp_files, test_size=0.5, random_state=cfg.seed)
    train_files, val_files, test_files = file_list[0:2], file_list[2:3], file_list[3:4]

    cfg["train_dataset_size"] = len(train_files)
    cfg["val_dataset_size"] = len(val_files)
    cfg["test_dataset_size"] = len(test_files)

    train_dataloader = DataLoader(GraphDataset(train_files, device=device), batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(GraphDataset(val_files,   device=device), batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(GraphDataset(test_files,  device=device), batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def main(cfg, debug):
    # Setup
    train_dl, val_dl, model, optimizer, scheduler, scaler, \
        start_epoch, best_acc, run_dir, writer, _ = \
        setup_training(
            cfg,
            create_model_fn=GraMIModel,
            data_loader_fn=load_graph_data,
        )

    # Loss + acc
    if not model.is_variational:
        loss_fn = lambda x, x_tile, adj_mat, n_V, n_A, z_A, z_V, eps_A, eps_V, edge_logits, x_tile_rec, x_rec, epoch: \
            GraMI_loss(x, x_tile, adj_mat, n_V, n_A, edge_logits,
                    x_tile_rec, x_rec,
                    variational=model.is_variational,
                    lambdas=cfg.loss_config["lambdas"],
                    betas=cfg.loss_config["betas"])
    else:
        loss_fn = lambda x, x_tile, adj_mat, n_V, n_A, z_A, z_V, eps_A, eps_V, edge_logits, x_tile_rec, x_rec, epoch: \
            SeeR_GraMI_loss(x, x_tile, adj_mat, n_V, n_A, z_A, z_V, eps_A, eps_V, edge_logits, x_tile_rec, x_rec, epoch, lambdas=cfg.loss_config["lambdas"])

    acc_fn = edge_and_r2_acc

    # Single step
    single_step_fn = lambda model, data, loss_fn, acc_fn, epoch, data_index: \
        single_step(model, data, loss_fn, acc_fn, writer, epoch, data_index, debug=debug)

    # Loop
    training_loop(cfg, debug, train_dl, val_dl, model, 
                  optimizer, scheduler, scaler,
                  start_epoch, best_acc, run_dir, writer, 
                  loss_fn, acc_fn, single_step_fn, log_training_metrics)

def single_step(model: GraMIModel, data, loss_fn, acc_fn, writer, epoch, data_index, debug) -> tuple[torch.Tensor, float]:
    adj_mat = get_adj_mat_from_edge_index(data.x_dict, data.edge_index_dict)

    x, x_tile, n_A, n_V, z_A, z_V, eps_A, eps_V, edge_logits, x_tile_rec, x_rec = model(data)

    if model.training and debug:
        for k, v in n_V.items():
            writer.add_histogram(f"latent/z_V_{k}_d{data_index}", v, epoch)
        writer.add_histogram(f"latent/z_A_d{data_index}", n_A, epoch)
        for k, v in edge_logits.items():
            writer.add_histogram(f"latent/edges_{k}_d{data_index}", v, epoch)

    assert all([x_tile[k].shape == x_tile_rec[k].shape[1:] for k in x_tile_rec.keys()]), \
        f"{[(k, x_tile[k].shape, x_tile_rec[k].shape) for k in x_tile_rec.keys() if x_tile[k].shape != x_tile_rec[k].shape]}"
    assert all([x[k].shape == x_rec[k].shape[1:] for k in x_rec.keys()]), \
        f"{[(k, x[k].shape, x_rec[k].shape) for k in x_rec.keys() if x[k].shape != x_rec[k].shape]}"
    assert all([adj_mat[k].shape == edge_logits[k].shape[1:] for k in data.edge_index_dict.keys()]), \
        f"{[(k, adj_mat[k].shape, edge_logits[k].shape) for k in data.edge_index_dict.keys() if adj_mat[k].shape != edge_logits[k].shape]}"

    # print(edge_logits)
    loss = loss_fn(x, x_tile, adj_mat, n_V, n_A, z_A, z_V, eps_A, eps_V, edge_logits, x_tile_rec, x_rec, epoch)
    with torch.no_grad():
        acc = acc_fn(x, adj_mat, edge_logits, x_rec)
    return loss, acc

def log_training_metrics(writer, i, mean_train_loss, mean_train_acc, mean_val_loss, mean_val_acc, lr):
    writer.add_scalar("Loss/train", mean_train_loss, i)
    writer.add_scalar("Edge Acc/train", mean_train_acc[0], i)
    writer.add_scalar("R2 Acc/train", mean_train_acc[1], i)
    writer.add_scalar("Loss/val", mean_val_loss, i)
    writer.add_scalar("Edge Acc/val", mean_val_acc[0], i)
    writer.add_scalar("R2 Acc/val", mean_val_acc[1], i)
    writer.add_scalar("lr", lr, i)

    print(f"{i:>4d}     |  Train  |  Valid  |")
    print(f"Loss     | {mean_train_loss:7.4f} | {mean_val_loss:7.4f} |")
    print(f"Edge Acc | {mean_train_acc[0]:7.4f} | {mean_val_acc[0]:7.4f} |")
    print(f"R2 Acc   | {mean_train_acc[1]:7.4f} | {mean_val_acc[1]:7.4f} |")

    writer.flush()


if __name__ == "__main__":
    extra_args = {
        "--loss_lambdas": dict(type=str, default=None),
        "--loss_betas": dict(type=str, default=None),
        "--ratio": dict(type=float, default=None),
    }

    def handle_loss(val, args, cfg, key):
        args.__dict__[key] = json.loads(val)
        loss_key = key.split("_")[1]
        print(f"Overriding config value loss_config[{loss_key}] with {val}")
        cfg.__dict__["loss_config"][loss_key] = args.__dict__[key]

    def handle_ratio(val, args, cfg, key):
        args.__dict__[key] = val
        cfg.__dict__["extra_config"][key] = val
        print(f"Adding config value extra_config[{key}] with {val}")

    extra_handlers = {
        "loss_lambdas": lambda v, a, c: handle_loss(v, a, c, "loss_lambdas"),
        "loss_betas": lambda v, a, c: handle_loss(v, a, c, "loss_betas"),
        "ratio": lambda v, a, c: handle_ratio(v, a, c, "ratio"),
    }

    parse_and_run(
        config_class=TrainConfig,
        main_fn=main,
        arg_definitions=extra_args,
        special_handlers=extra_handlers,
    )
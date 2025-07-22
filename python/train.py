import sys
import json
import glob
import re
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset import HecBenchDataset
from utils.common import (
    get_adj_mat_from_edge_index,
    device,
    epochs,
    get_log_dir_name,
    train_from_checkpoint,
    lr,
    decay,
    batch_size,
    get_data_shape,
)
from models.GraMI.metrics import loss_fn, acc_fn
from models.GraMI import GraMI
from paths import GraMI_path, top_level_path


file_list = list((top_level_path / "HecBench" / "heterodatas").glob("*.pt"))
train_files, temp_files = train_test_split(file_list, test_size=0.4, random_state=42)
val_files,   test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
print(f"Training/Validation/Test split: {len(train_files)}/{len(val_files)}/{len(test_files)}")


import glob, re

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

def single_step(data, model):
    adj_mat = get_adj_mat_from_edge_index(data.x_dict, data.edge_index_dict)

    X, X_hat, A, V, edge_logits, X_hat_prime, X_prime = model(data)

    assert all([X_hat[k].shape == X_hat_prime[k].shape for k in X_hat_prime.keys()]), \
        f"{[(k, X_hat[k].shape, X_hat_prime[k].shape) for k in X_hat_prime.keys() if X_hat[k].shape != X_hat_prime[k].shape]}"
    assert all([X[k].shape == X_prime[k].shape for k in X_prime.keys()]), \
        f"{[(k, X[k].shape, X_prime[k].shape) for k in X_prime.keys() if X[k].shape != X_prime[k].shape]}"
    assert all([adj_mat[k].shape == edge_logits[k].shape for k in data.edge_index_dict.keys()]), \
        f"{[(k, adj_mat[k].shape, edge_logits[k].shape) for k in data.edge_index_dict.keys() if adj_mat[k].shape != edge_logits[k].shape]}"

    loss = loss_fn(X, X_hat, adj_mat, V, A, edge_logits, X_hat_prime, X_prime)
    with torch.no_grad():
        edge_acc, r2_attr = acc_fn(X, adj_mat, edge_logits, X_prime)
    return loss, edge_acc, r2_attr 

def main():
    train_dataloader = DataLoader(HecBenchDataset(train_files, device=device), batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(HecBenchDataset(val_files, device=device), batch_size=batch_size, shuffle=False)

    data_sample = next(iter(train_dataloader))
    
    with open(GraMI_path / "config.json") as f:
        model_config = json.load(f)

    model = GraMI(model_config, get_data_shape(data_sample), device, batch_size)

    model_name = model_config["model_name"]
    print(model_name)

    e, latest = find_latest_file(str(GraMI_path / f"{model_name}_")+"*.pt")
    if train_from_checkpoint and latest is not None and Path(latest).exists():
        # with torch.serialization.safe_globals([torch.nn.parameter.UninitializedParameter]):
            model.load_state_dict(torch.load(latest), strict=True)

    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    writer = SummaryWriter(log_dir=get_log_dir_name(model_config["model_name"]))

    for i in range(e, epochs):
        print("Epoch:", i)
        index_train = 0
        tot_train_loss = 0
        tot_train_edge_acc = 0
        tot_train_r2_attr = 0

        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            loss, edge_acc, r2_attr = single_step(batch, model)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item() * batch.batch_size
            tot_train_edge_acc += edge_acc.item() * batch.batch_size
            tot_train_r2_attr += r2_attr.item() * batch.batch_size
            index_train += batch.batch_size


        index_val = 0
        tot_val_loss = 0
        tot_val_edge_acc = 0
        tot_val_r2_attr = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                loss, edge_acc, r2_attr = single_step(batch, model)
                tot_val_loss += loss.item() * batch.batch_size
                tot_val_edge_acc += edge_acc.item() * batch.batch_size
                tot_val_r2_attr += r2_attr.item() * batch.batch_size
                index_val += batch.batch_size

        if i % 100 == 0:
            torch.save(model.state_dict(), GraMI_path / f"{model_name}_{i}.pt")

        writer.add_scalar("Loss/train", tot_train_loss / index_train, i)
        writer.add_scalar("Acc/train", tot_train_edge_acc / index_train, i)
        writer.add_scalar("Acc/train", tot_train_r2_attr / index_train, i)
        writer.add_scalar("Loss/val", tot_val_loss / index_val, i)
        writer.add_scalar("edge-acc/val", tot_val_edge_acc / index_val, i)
        writer.add_scalar("r2-attr/val", tot_val_r2_attr / index_val, i)

        print(f"Epoch:          {i}")
        print(f"Loss/train:     {tot_train_loss / index_train}")
        print(f"edge-acc/train: {tot_train_edge_acc / index_train}")
        print(f"r2-attr/train:  {tot_train_r2_attr / index_train}")
        print(f"Loss/val:       {tot_val_loss / index_val}")
        print(f"edge-acc/val:   {tot_val_edge_acc / index_val}")
        print(f"r2-attr/val:    {tot_val_r2_attr / index_val}")


        writer.flush()
    
    writer.close()

if __name__ == "__main__":
    # python train.py --d_embed 10 --init_instr "64,32" --init_val "64,32" --init_num "" --init_typ "32" --init_attr "" --init_size "" --init_fdim 16 --enc_hgnn "16,16" --enc_mlp "64,16" --enc_fdim 8 --mname "latest" 
    main()
    print("Training complete.")

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader

from utils.paths import top_level_path, runs_dir
from dataset import DevmapDataset
from utils.common import (
    find_latest_run_dir, 
    find_latest_wgts, 
    get_data_shape, 
    get_log_dir_name, 
    copy_model_arch_to_dir, 
    copy_config_to_dir,
    set_seed,
    get_timestamp
)
from utils.config import cfg, load_config, configs_dir
from models.Devmap import DevmapModel

os.environ["HF_HOME"] = str(top_level_path.parent / "hf")

def single_step(model, batch, labels, loss_fn):
    logits = model(batch)
    probs = torch.sigmoid(logits)
    loss = loss_fn(probs, labels.to(torch.float32)).mean()
    acc = ((probs >= 0.5).to(torch.int32) == labels).float().mean()
    return loss, acc

def load_data(dataset, device, batch_size, seed=42):
    ################ Load data ################
    dataset_dir = top_level_path / dataset
    assert dataset_dir.exists() and dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist. Please check the dataset name: {dataset}"
    data_path = dataset_dir / "heterodatas"
    assert data_path.exists() and data_path.is_dir(), f"Data path {data_path} does not exist. Please check the data directory."
    csv_file  = data_path / "datapoints.csv"
    assert csv_file.exists(), f"CSV file {csv_file} does not exist. Please check the file path."

    with open(csv_file) as f:
        df = pd.read_csv(f)

    df["file_path"] = df["pt_file"].apply(lambda x: str(data_path / x))
    file_list = df["file_path"].tolist()
    devmap_list = df["device"].tolist()[:len(file_list)]

    train_files, temp_files, train_devmap, temp_devmap = train_test_split(file_list,  devmap_list, test_size=0.4, random_state=seed)
    val_files,   test_files, val_devmap,   test_devmap = train_test_split(temp_files, temp_devmap, test_size=0.5, random_state=seed)

    train_dataloader = DataLoader(DevmapDataset(train_files, train_devmap, device=device), batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(DevmapDataset(val_files,   val_devmap,   device=device), batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(DevmapDataset(test_files,  test_devmap,  device=device), batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def load_model(model_name, data_shapes, cfg):
    device, batch_size, train_from_checkpoint = cfg.device, cfg.batch_size, cfg.train_from_checkpoint

    run_dir = None
    pretrained_weights_file = None
    if train_from_checkpoint:
        run_dir = find_latest_run_dir(model_name)

    if run_dir is not None: 
        # Found existing run directory
        assert run_dir.is_dir(), f"Expected run_dir ({run_dir}) to be a folder"
        model_arch_file = run_dir / f"{model_name}.json"
        pretrained_weights_file = find_latest_wgts(run_dir, model_name)
    else:
        # New training run
        run_dir = runs_dir / get_log_dir_name(model_name)
        model_arch_file = configs_dir / f"{model_name}.json"
        copy_model_arch_to_dir(model_arch_file, run_dir)

    cfg.save(run_dir / f"config_{get_timestamp()}.yaml")

    with open(model_arch_file) as f:
        Devmap_config = json.load(f)

    model_config = Devmap_config

    model = DevmapModel(model_config, data_shapes, device, batch_size)

    print(model_name)


    start_epoch = 0
    if pretrained_weights_file is not None and Path(pretrained_weights_file).exists():
        # with torch.serialization.safe_globals([torch.nn.parameter.UninitializedParameter]):
        model.load_state_dict(torch.load(pretrained_weights_file), strict=True)
        
        # Extract epoch number from the checkpoint filename, e.g., "modelname_123.pt"
        try:
            start_epoch = int(pretrained_weights_file.stem.split("_")[-1])
        except (ValueError, AttributeError):
            pass

    model.to(device)

    return model, start_epoch, run_dir

def main(cfg):
    device, model_name, dataset, batch_size, epochs, lr, decay = (
        cfg.device, cfg.model_name, cfg.dataset, cfg.batch_size,  # type: ignore
        cfg.epochs, cfg.learning_rate, cfg.weight_decay)          # type: ignore


    train_dataloader, val_dataloader, test_dataloader = load_data(dataset, device, batch_size, seed=cfg.seed)
    data_sample, label = next(iter(train_dataloader))
    data_shapes = get_data_shape(data_sample)

    model, start_epoch, run_dir = load_model(model_name, data_shapes, cfg)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    writer = SummaryWriter(log_dir=run_dir)

    for i in range(start_epoch, epochs):
        index_train = 0
        tot_train_loss = 0
        tot_train_acc  = 0

        model.train()
        for batch, labels in tqdm(train_dataloader, desc=f"Train {i}"):
            optimizer.zero_grad()
            loss, acc = single_step(model, batch, labels, criterion)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item() * batch.batch_size
            tot_train_acc  += acc.item() * batch.batch_size
            index_train    += batch.batch_size


        index_val = 0
        tot_val_loss = 0
        tot_val_acc = 0

        model.eval()
        with torch.no_grad():
            for batch, labels in tqdm(val_dataloader, desc=f"Valid {i}"):
                loss, acc = single_step(model, batch, labels, criterion)
                tot_val_loss += loss.item() * batch.batch_size
                tot_val_acc  += acc.item() * batch.batch_size
                index_val    += batch.batch_size

        if i % 10 == 0 and i != 0:
            torch.save(model.state_dict(), run_dir / f"{model_name}_{i}.pt")

        writer.add_scalar("Loss/train", tot_train_loss / index_train, i)
        writer.add_scalar("Acc/train", tot_train_acc / index_train, i)
        writer.add_scalar("Loss/val", tot_val_loss / index_val, i)
        writer.add_scalar("Acc/val", tot_val_acc / index_val, i)

        print(f"     |  Train  |  Valid  |")
        print(f"Loss | {tot_train_loss / index_train:7.4f} | {tot_val_loss / index_val:7.4f} |")
        print(f"Acc  | {tot_train_acc / index_train:7.4f} | {tot_val_acc / index_val:7.4f} |")

        writer.flush()

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()


    # if args.config:
    cfg = load_config(configs_dir / f"{args.config}.yaml", flatten=True)

    set_seed(cfg.seed)

    main(cfg)
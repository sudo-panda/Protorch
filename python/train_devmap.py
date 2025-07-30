import argparse
import pickle
import os
import json
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split

import random
import numpy as np
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
    copy_file_to_dir,
    make_deterministic,
    get_timestamp,
    find_latest_file
)
from utils.train import get_scheduler_fn, get_scheduler_step_type
from utils.config import TrainConfig, load_config, configs_dir
from models.Devmap import DevmapModel

os.environ["HF_HOME"] = str(top_level_path.parent / "hf")

def single_step(model, batch, labels, loss_fn):
    logits = model(batch)
    probs = torch.sigmoid(logits)
    loss = loss_fn(probs, labels.to(torch.float32)).mean()
    acc = ((probs >= 0.5).to(torch.int32) == labels).float().mean()
    return loss, acc

def load_data(dataset, device, batch_size, cfg):
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

    train_files, temp_files, train_devmap, temp_devmap = train_test_split(file_list,  devmap_list, test_size=0.4, random_state=cfg.seed)
    val_files,   test_files, val_devmap,   test_devmap = train_test_split(temp_files, temp_devmap, test_size=0.5, random_state=cfg.seed)

    cfg["train_dataset_size"] = len(train_files)
    cfg["val_dataset_size"] = len(val_files)
    cfg["test_dataset_size"] = len(test_files)

    train_dataloader = DataLoader(DevmapDataset(train_files, train_devmap, device=device), batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(DevmapDataset(val_files,   val_devmap,   device=device), batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(DevmapDataset(test_files,  test_devmap,  device=device), batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def load_model(model_name, data_shapes, cfg):
    train_from_checkpoint = cfg.train_from_checkpoint
    run_dir = runs_dir / get_log_dir_name(model_name)

    prev_run_dir = None
    if isinstance(train_from_checkpoint, bool) and train_from_checkpoint == True:
        prev_run_dir = find_latest_run_dir(model_name)
    elif isinstance(train_from_checkpoint, str):
        prev_run_dir = runs_dir / train_from_checkpoint
        assert prev_run_dir.exists(), f"Run directory does not exist:\n\t{prev_run_dir}\n\tPlease give the right checkpoint directory name in train_from_checkpoint"

    save_file, prev_cfg = None, None
    if prev_run_dir is not None: 
        # Found existing run directory
        assert prev_run_dir.is_dir(), f"Expected run_dir ({prev_run_dir}) to be a folder"

        model_arch_file = prev_run_dir / f"{model_name}.json"
        print(f"Found existing run directory:\n\t{prev_run_dir}\n\twith model architecture file {model_arch_file.name}")
        save_file = find_latest_wgts(prev_run_dir, model_name)
        copy_file_to_dir(model_arch_file, run_dir)

        prev_config_file = find_latest_file(prev_run_dir, "config*.yaml")
        assert prev_config_file is not None, f"Config file not found in prev run dir:\n\t{prev_run_dir}"

        prev_cfg = load_config(prev_config_file, train=True)
        assert isinstance(prev_cfg, TrainConfig), f"Previous config is not a TrainConfig: {prev_config_file}"

        prev_cfg.save(run_dir / f"prev-config_{prev_run_dir.name}.yaml")
    else:
        # New training run
        model_arch_file = configs_dir / f"{model_name}.json"

        print(f"Creating new run directory {run_dir} with model architecture file {model_arch_file.name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        copy_file_to_dir(model_arch_file, run_dir)


    with open(model_arch_file) as f:
        model_config = json.load(f)

    device, batch_size = cfg.device, cfg.batch_size
    model = DevmapModel(model_config, data_shapes, device, batch_size)

    criterion = nn.BCELoss()

    optimizer = create_optimizer(cfg, model)

    scheduler = create_scheduler(cfg, optimizer)
    scaler = None

    valid_acc, start_epoch = restore_training_state(
        cfg, prev_cfg, save_file, model, optimizer, scheduler)

    model.to(device)

    cfg.save(run_dir / f"config.yaml")

    return model, optimizer, criterion, scheduler, scaler, start_epoch, valid_acc, run_dir

def create_optimizer(cfg, model):
    opt_name, lr, decay = cfg.optimizer, cfg.learning_rate, cfg.weight_decay
    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise NotImplementedError(f"Optimizer {opt_name} is not implemented")
    return optimizer

def restore_training_state(cfg, prev_cfg, save_file, model, optimizer, scheduler):
    opt_name, lr, decay, device = cfg.optimizer, cfg.learning_rate, cfg.weight_decay, cfg.device
    valid_acc = None
    start_epoch = 0
    if save_file is not None and save_file.exists():
        print(f"Loading pretrained weights from {save_file.name}")
        try:
            saved_state = torch.load(save_file, map_location=device)
        except pickle.UnpicklingError:
            saved_state = torch.load(save_file, map_location=device, weights_only=False)

        pretrained_weights_file = saved_state["model"]
        model.load_state_dict(pretrained_weights_file, strict=True)

        assert prev_cfg.seed == cfg.seed, f"Previous seed {prev_cfg.seed} does not match current seed {cfg.seed}"

        if prev_cfg.learning_rate == lr and prev_cfg.weight_decay == decay and prev_cfg.optimizer == opt_name:
            optimizer_state = saved_state["optimizer"]
                # Load optimizer state
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            if scheduler and prev_cfg.scheduler == cfg.scheduler and saved_state["scheduler"]:
                scheduler.load_state_dict(saved_state["scheduler"])
        else:
            print(f"Warning: Optimizer args in previous config differs from current config\n"
                    f"   previous: {prev_cfg.optimizer}, {prev_cfg.learning_rate}, {prev_cfg.weight_decay}\n"
                    f"   current: {opt_name}, {lr}, {decay}\n"
                    f"Creating new optimizer state . . .")

        start_epoch = saved_state["epoch"] + 1
        valid_acc = saved_state["valid_acc"]

        torch.set_rng_state(saved_state["rng_state"]["torch"].clone().type(torch.ByteTensor))
        torch.cuda.set_rng_state_all([s.clone().type(torch.ByteTensor) for s in saved_state["rng_state"]["cuda"]])
        if "numpy_seed" in saved_state["rng_state"]:
            np.random.seed(saved_state["rng_state"]["numpy_seed"])
            random.seed(saved_state["rng_state"]["python_seed"])
        else:
            np.random.set_state(saved_state["rng_state"]["numpy"])
            random.setstate(saved_state["rng_state"]["python"])
    return valid_acc, start_epoch

def create_scheduler(cfg, optimizer):
    scheduler = None
    if cfg.scheduler is not None:
        assert cfg.scheduler.get("name") is not None, "Scheduler name must be provided in the config"
        scheduler_name = cfg.scheduler["name"]
        del cfg.scheduler["name"]

        total_training_steps = cfg.epochs * ((cfg.train_dataset_size - 1) // cfg.batch_size + 1)
        if cfg.scheduler.get("num_warmup_steps") is not None:
            num_warmup_steps = cfg.scheduler["num_warmup_steps"]
            del cfg.scheduler["num_warmup_steps"]
        elif cfg.scheduler.get("warmup_ratio") is not None:
            num_warmup_steps = int(cfg.scheduler["warmup_ratio"] * total_training_steps)
            del cfg.scheduler["warmup_ratio"]
        else:
            # Default warmup ratio of 5%
            num_warmup_steps = int(total_training_steps * 0.05)
        
        scheduler = get_scheduler_fn(
            scheduler_name,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            **cfg.scheduler
        )
        
    return scheduler

def main(cfg):
    device, model_name, dataset, batch_size, epochs, lr, decay = (
        cfg.device, cfg.model_name, cfg.dataset, cfg.batch_size,  # type: ignore
        cfg.epochs, cfg.learning_rate, cfg.weight_decay)          # type: ignore

    print(f"Training {model_name} on {dataset}")
    print(f"  epochs: {epochs}\n  batch size: {batch_size}\n  learning rate: {lr}\n  weight decay: {decay}")

    train_dataloader, val_dataloader, _ = load_data(dataset, device, batch_size, cfg)
    data_sample, _ = next(iter(train_dataloader))
    data_shapes = get_data_shape(data_sample)

    model, optimizer, criterion, scheduler, scaler, start_epoch, best_valid_acc, run_dir = load_model(model_name, data_shapes, cfg)
    writer = SummaryWriter(log_dir=run_dir)
    if best_valid_acc is not None:
        print(f"  Best Valid Acc: {best_valid_acc}")

    scheduler_step_type = get_scheduler_step_type(scheduler)

    for i in range(start_epoch, epochs):
        mean_train_loss, mean_train_acc = train_one_epoch(train_dataloader, model, optimizer, criterion, scheduler, scheduler_step_type, i)

        mean_val_loss, mean_val_acc = validate_model(val_dataloader, model, criterion, i)

        if scheduler_step_type == "metric_min":
            scheduler.step(mean_val_loss)
        elif scheduler_step_type == "metric_max":
            scheduler.step(mean_val_acc)

        if best_valid_acc is None or mean_val_acc > best_valid_acc:
            print("Improved validation accuracy! Saving ...", end="\t", flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),  # optional if changing later
                    "epoch": i,
                    "valid_acc": mean_val_acc,
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "rng_state": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                        "numpy_seed": np.random.get_state()[1][0],
                        "python_seed": random.getstate()[1][0],
                    }
                },
                run_dir / f"{model_name}_{get_timestamp()}.pt"
            )

            best_valid_acc = mean_val_acc
            print("Done", flush=True)

        log_training_metrics(writer, i, mean_train_loss, mean_train_acc, 
                             mean_val_loss, mean_val_acc, 
                             get_current_lr(optimizer, scheduler))

    writer.close()

def train_one_epoch(train_dataloader, model, optimizer, criterion, scheduler, scheduler_step_type, i):
    index_train = 0
    tot_train_loss = 0
    tot_train_acc  = 0

    model.train()
    for batch, labels in tqdm(train_dataloader, desc=f"Train {i}"):
        optimizer.zero_grad()
        loss, acc = single_step(model, batch, labels, criterion)
        loss.backward()
        optimizer.step()
        if scheduler_step_type == "batch":
            scheduler.step()
        tot_train_loss += loss.item() * batch.batch_size
        tot_train_acc  += acc.item() * batch.batch_size
        index_train    += batch.batch_size

    mean_train_loss = tot_train_loss / index_train
    mean_train_acc  = tot_train_acc / index_train

    if scheduler_step_type == "epoch":
            scheduler.step()
    
    return mean_train_loss,mean_train_acc

def validate_model(val_dataloader, model, criterion, i):
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

    mean_val_loss   = tot_val_loss / index_val
    mean_val_acc    = tot_val_acc / index_val
    return mean_val_loss,mean_val_acc

def log_training_metrics(writer, i, mean_train_loss, mean_train_acc, mean_val_loss, mean_val_acc, lr):
    writer.add_scalar("Loss/train", mean_train_loss, i)
    writer.add_scalar("Acc/train", mean_train_acc, i)
    writer.add_scalar("Loss/val", mean_val_loss, i)
    writer.add_scalar("Acc/val", mean_val_acc, i)
    writer.add_scalar("lr", lr, i)

    print(f"{i:>4d} |  Train  |  Valid  |")
    print(f"Loss | {mean_train_loss:7.4f} | {mean_val_loss:7.4f} |")
    print(f"Acc  | {mean_train_acc:7.4f} | {mean_val_acc:7.4f} |")

    writer.flush()

def get_current_lr(optimizer, scheduler):
    lr = None
    if scheduler:
        lr = scheduler.get_last_lr()[0]
    else:
        lr = optimizer.param_groups[0]["lr"]
    return lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()


    # if args.config:
    config_path = configs_dir / f"{args.config}.yaml"
    print(f"Loading config from {config_path}")
    assert config_path.exists(), f"Config file {config_path} does not exist. Please check the config name."

    cfg = load_config(config_path, train=True)
    assert isinstance(cfg, TrainConfig), f"Config loaded is not a TrainConfig, got {type(cfg)}"

    make_deterministic(cfg.seed)

    main(cfg)
    print(f"Training Done!")
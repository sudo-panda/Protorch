import torch
import torch.nn as nn
import json
from pathlib import Path

from utils.config import TrainConfig, load_config
from utils.common import (
    find_latest_run_dir, 
    find_latest_wgts,
    find_latest_file,
)

from utils.train import (
    log_config,
    log_model_arch,
    parse_and_run,
    setup_training,
    training_loop,
)

import pickle
from train_devmap import load_devmap_data

from models.GraMI import GraMIModel

from models.Devmap.model import DevmapClassifier

from torch.utils.tensorboard import SummaryWriter


def load_GraMI_encoder(model_name: str, data_shapes: dict, device: str, writer: SummaryWriter, load_run_dir=None, save_file=None):
    if load_run_dir is None:
        load_run_dir = find_latest_run_dir(model_name)
    assert load_run_dir is not None, f"Run directory not found for model: {model_name}"
    model_arch_file = load_run_dir / f"{model_name}.json"

    print(f"Found GraMI run directory:\n\t{load_run_dir}\n\twith model architecture file {model_arch_file.name}")

    if save_file is None:
        save_file = find_latest_wgts(load_run_dir, model_name)
    assert save_file is not None and save_file.exists(), f"Weights file not found in run dir:\n\t{load_run_dir}"

    prev_config_file = find_latest_file(load_run_dir, "config*.yaml")
    assert prev_config_file is not None, f"Config file not found in prev run dir:\n\t{load_run_dir}"

    prev_cfg = load_config(prev_config_file, train=True)

    with open(model_arch_file) as f:
        model_arch = json.load(f)

    log_model_arch(writer, model_arch, tag="GraMI_arch")
    log_config(writer, prev_cfg, tag="GraMI_config")

    model = GraMIModel(model_arch, data_shapes)
    model.to(device)

    try:
        saved_state = torch.load(save_file, map_location=device)
    except pickle.UnpicklingError:
        saved_state = torch.load(save_file, map_location=device, weights_only=False)

    pretrained_weights_file = saved_state["model"]
    model.load_state_dict(pretrained_weights_file, strict=True)

    return model.encoder

def load_grami_enc(model_config: dict, data_shapes: dict, model_arch_file: Path, device: str, writer: SummaryWriter):
    GraMI_config = model_config["GraMI"]
    if "save_file" in GraMI_config:
        GraMI_save_file = Path(GraMI_config["save_file"]).absolute()
        assert GraMI_save_file.exists(), f"GraMI save file does not exist:\n\t{str(GraMI_save_file)}\n Check the model arch file: {str(model_arch_file)}"
        assert GraMI_save_file.is_file(), f"GraMI save file is not a file:\n\t{str(GraMI_save_file)}\n Check the model arch file: {str(model_arch_file)}"
        GraMI_run_dir = GraMI_save_file.parent.absolute()
        assert GraMI_run_dir.exists() and GraMI_run_dir.is_dir(), f"GraMI run directory does not exist:\n\t{str(GraMI_run_dir)}\n Check the model arch file:{str(model_arch_file)}"
    elif "run_dir" in GraMI_config:
        GraMI_run_dir = Path(GraMI_config["run_dir"]).absolute()
        assert GraMI_run_dir.exists() and GraMI_run_dir.is_dir(), f"GraMI run directory does not exist:\n\t{str(GraMI_run_dir)}\n Check the model arch file:{str(model_arch_file)}"
        GraMI_save_file = None
    else:
        GraMI_run_dir = None
        GraMI_save_file = None

    grami_enc = load_GraMI_encoder(GraMI_config["arch"], data_shapes, device,
                                   writer, load_run_dir=GraMI_run_dir,
                                    save_file=GraMI_save_file)

    output_shape = grami_enc.get_output_shape(data_shapes)

    data_shapes = {}
    if grami_enc.variational:
        data_shapes["z_A"] = output_shape["n_A"][0]
        data_shapes["z_V"] = {k: v[0] for k, v in output_shape["n_V"].items()}
    else:
        data_shapes["z_A"] = output_shape["n_A"]
        data_shapes["z_V"] = output_shape["n_V"]
    
    grami_enc.to(device)

    def encoder(batch):
        _, _, n_A, n_V = grami_enc(batch)
        if grami_enc.variational:
            z_A = n_A[0]
            z_V = n_V[0]
        else:
            z_A = n_A
            z_V = n_V

        return z_A, z_V
    
    return data_shapes, encoder


def main(cfg, debug):
    # Setup
    train_dl, val_dl, model, optimizer, scheduler, scaler, start_epoch, \
        best_acc, run_dir, writer, GraMI_enc = \
        setup_training(
            cfg,
            create_model_fn=lambda model_config, data_shapes, extra_config: DevmapClassifier(model_config["classifier"], data_shapes, extra_config=extra_config),
            data_loader_fn=load_devmap_data,
            load_preprocessor_fn=load_grami_enc,
        )

    # Loss + acc
    loss_fn = nn.BCELoss()
    acc_fn = lambda preds, labels: ((preds >= 0.5).to(torch.int32) == labels).type(torch.float32)

    # Single step
    single_step_fn = lambda model, data, loss_fn, acc_fn, epoch, data_index: \
        single_step(model, data[0], data[1], loss_fn, acc_fn, GraMI_enc)

    # Loop
    training_loop(cfg, debug, train_dl, val_dl, model, optimizer, 
                  scheduler, scaler, start_epoch, best_acc, run_dir, 
                  writer, loss_fn, acc_fn, single_step_fn, log_training_metrics)

def single_step(model, batch, labels, loss_fn, acc_fn, GraMI_enc):
    z_A, z_V = GraMI_enc(batch)
    assert isinstance(model, DevmapClassifier)
    logits = model(
        z_A.detach(), {k: v.detach() for k, v in z_V.items()}, 
        batch.comp, batch.mem, batch.localmem, 
        batch.coalesced, batch.transfer, batch.wgsize, 
        {k: batch[k].batch for k in z_V.keys()}
    )
    probs = torch.sigmoid(logits)
    loss = loss_fn(probs, labels.to(torch.float32)).mean()
    acc = acc_fn(probs, labels).mean()
    return loss, acc

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


if __name__ == "__main__":
    parse_and_run(
        config_class=TrainConfig,
        main_fn=main,
    )
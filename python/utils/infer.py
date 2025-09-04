import json
import pickle
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
from utils.config import TestConfig, TrainConfig, load_config
from utils.common import (
    find_latest_run_dir, 
    find_latest_wgts,
    find_latest_file,
    get_data_shape,
)
from utils.train import (
    log_config,
    log_model_arch,
)

from models.GraMI import GraMIModel

from torch.utils.tensorboard import SummaryWriter

def load_pretrained_grami_model(model_name: str, data_shapes: dict, device: str, writer: Optional[SummaryWriter], load_run_dir=None, save_file=None):
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

    return model

def load_grami_enc(model_config: dict,
                   data_shapes: dict,
                   model_arch_file: Path,
                   device: str,
                   writer: Optional[SummaryWriter]):
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

    grami = load_pretrained_grami_model(GraMI_config["arch"], data_shapes, device,
                                        writer, load_run_dir=GraMI_run_dir,
                                        save_file=GraMI_save_file)
    grami_enc = grami.encoder

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


def load_testing_modules(
        create_model_fn, 
        model_name: str, 
        data_shapes: dict, 
        cfg: TestConfig, 
        load_preprocessor: Optional[Callable[[dict, dict, Path, str, Optional[SummaryWriter]], tuple]] = None):
    checkpoint = cfg.checkpoint
    print(f"Using checkpoint: {checkpoint}")

    save_file, prev_run_dir, prev_cfg = None, None, None
    if checkpoint is None:
        prev_run_dir = find_latest_run_dir(model_name)
    elif isinstance(checkpoint, str):
        save_file = Path(checkpoint).absolute()
        assert save_file.exists() and save_file.is_file(), f"Checkpoint file does not exist or is not a file:\n\t{save_file}\n  Please give the right checkpoint file name in train_from_checkpoint"
        prev_run_dir = save_file.parent
        assert prev_run_dir.exists() and prev_run_dir.is_dir(), f"Run directory does not exist:\n\t{prev_run_dir}\n\tPlease give the right checkpoint directory name in train_from_checkpoint"

    assert (prev_run_dir is not None), f"The prev_run_dir var not set. Please provide a checkpoint or ensure that a previous run exists for model {model_name}."
    # Found existing run directory
    assert prev_run_dir.is_dir(), f"Expected run_dir ({prev_run_dir}) to be a folder"

    model_arch_file = prev_run_dir / f"{model_name}.json"
    print(f"Found existing run directory:\n\t{prev_run_dir}\n\twith model architecture file {model_arch_file.name}")
    if save_file is None:
        save_file = find_latest_file(prev_run_dir, f"{model_name}_*.pt")
        assert save_file is not None and save_file.exists(), f"Weight file not found in prev run dir:\n\t{prev_run_dir}"
    
    csv_file_name = save_file.with_suffix(".csv")
    print(f"\n\tWriting predictions to {csv_file_name}\n")

    prev_config_file = find_latest_file(prev_run_dir, "config*.yaml")
    assert prev_config_file is not None, f"Config file not found in prev run dir:\n\t{prev_run_dir}"

    prev_cfg = load_config(prev_config_file, train=True)
    assert isinstance(prev_cfg, TrainConfig), f"Previous config is not a TrainConfig: {prev_config_file}"

    with open(model_arch_file) as f:
        model_config : dict = json.load(f)

    device : str = cfg.device
    preprocessor = None
    if load_preprocessor is not None:
        data_shapes, preprocessor = load_preprocessor(
            model_config, data_shapes, model_arch_file, device, None)

    model = create_model_fn(model_config, data_shapes, prev_cfg.extra_config)
    
    device = cfg.device
    
    if save_file is not None and save_file.exists():
        print(f"Loading pretrained weights from {save_file.name}")
        try:
            saved_state = torch.load(save_file, map_location=device)
        except pickle.UnpicklingError:
            saved_state = torch.load(save_file, map_location=device, weights_only=False)

        pretrained_weights_file = saved_state["model"]
        model.load_state_dict(pretrained_weights_file, strict=True)

        if prev_cfg is not None:
            assert prev_cfg.seed == cfg.seed, f"Previous seed {prev_cfg.seed} does not match current seed {cfg.seed}"

        # Restore RNG states
        torch.set_rng_state(saved_state["rng_state"]["torch"].clone().type(torch.ByteTensor))
        torch.cuda.set_rng_state_all([s.clone().type(torch.ByteTensor) for s in saved_state["rng_state"]["cuda"]])
        if "numpy_seed" in saved_state["rng_state"]:
            np.random.seed(saved_state["rng_state"]["numpy_seed"])
            random.seed(saved_state["rng_state"]["python_seed"])
        else:
            np.random.set_state(saved_state["rng_state"]["numpy"])
            random.setstate(saved_state["rng_state"]["python"])

    model.to(device)

    return model, prev_run_dir, csv_file_name, preprocessor

def setup_testing(
    cfg: TestConfig,
    create_model_fn,
    data_loader_fn: Callable[[str, str, int, TestConfig], tuple[DataLoader, DataLoader, DataLoader]],
    load_preprocessor_fn=None
) -> tuple[DataLoader, nn.Module, Path, Path, Optional[nn.Module]]:
    device, model_name, dataset, batch_size = (
        cfg.device, cfg.model_name, cfg.dataset, cfg.batch_size
    )
    _, _, test_dataloader = data_loader_fn(dataset, device, batch_size, cfg)
    sample = next(iter(test_dataloader))
    data_sample = sample[0] if isinstance(sample, (tuple, list)) else sample
    data_shapes = get_data_shape(data_sample)

    # --- Load model, optimizer, scheduler, scaler
    model, run_dir, csv_file_name, preprocessor = \
        load_testing_modules(create_model_fn, model_name, data_shapes, 
                             cfg, load_preprocessor=load_preprocessor_fn)

    return (test_dataloader, model, run_dir, csv_file_name, preprocessor)
import json
import pickle
import torch
from pathlib import Path
from utils.config import load_config
from utils.common import (
    find_latest_run_dir, 
    find_latest_wgts,
    find_latest_file,
)
from utils.train import (
    log_config,
    log_model_arch,
)

from models.GraMI import GraMIModel

from torch.utils.tensorboard import SummaryWriter

def load_pretrained_grami_model(model_name: str, data_shapes: dict, device: str, writer: SummaryWriter, load_run_dir=None, save_file=None):
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
import argparse
import gc
import json
from typing import Callable, Optional
import numpy as np
import torch
import pickle
import random
from pathlib import Path
from tqdm import tqdm
from transformers.optimization import get_scheduler as hf_get_scheduler
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau,
    OneCycleLR, CyclicLR, LambdaLR
)
from torch.utils.tensorboard import SummaryWriter

from dataclasses import asdict, is_dataclass
import yaml

from utils.common import (
    find_latest_run_dir,
    get_data_shape, 
    get_log_dir_name,
    find_latest_run_dir, 
    find_latest_wgts, 
    get_log_dir_name, 
    copy_file_to_dir,
    find_latest_file,
    make_deterministic,
    print_gpu_mem,
    sizeof_fmt,
)
from utils.config import TrainConfig, load_config
from utils.paths import configs_dir, runs_dir

def get_scheduler_fn(
    scheduler_name,
    optimizer,
    num_warmup_steps=0,
    num_training_steps=1000,
    last_epoch=-1,
    **kwargs
):
    scheduler_name = scheduler_name.lower()

    # HuggingFace schedulers
    hf_schedulers = {
        "linear", "cosine", "cosine_with_restarts",
        "polynomial", "constant", "constant_with_warmup"
    }

    if scheduler_name in hf_schedulers:
        print(f"Scheduler: {scheduler_name} Warmup Steps: {num_warmup_steps}, Total Steps: {num_training_steps}")
        return hf_get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=kwargs
        )

    # PyTorch-native schedulers
    elif scheduler_name == "steplr":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
            last_epoch=last_epoch
        )
    elif scheduler_name == "multisteplr":
        return MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [30, 80]),
            gamma=kwargs.get("gamma", 0.1),
            last_epoch=last_epoch
        )
    elif scheduler_name == "exponentiallr":
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.95),
            last_epoch=last_epoch
        )
    elif scheduler_name == "reducelronplateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            threshold=kwargs.get("threshold", 1e-4),
            min_lr=kwargs.get("min_lr", 0),
        )
    elif scheduler_name == "onecyclelr":
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 0.01),
            total_steps=num_training_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4)
        )
    elif scheduler_name == "cycliclr":
        return CyclicLR(
            optimizer,
            base_lr=kwargs.get("base_lr", 1e-5),
            max_lr=kwargs.get("max_lr", 1e-3),
            step_size_up=kwargs.get("step_size_up", 2000),
            mode=kwargs.get("mode", "triangular"),
            cycle_momentum=False
        )
    elif scheduler_name == "lambdalr":
        lambda_fn = kwargs.get("lr_lambda", lambda epoch: 1.0)
        return LambdaLR(optimizer, lr_lambda=lambda_fn)

    else:
        raise NotImplementedError(f"Scheduler '{scheduler_name}' is not implemented.")

def create_scheduler(scheduler_cfg, optimizer, total_training_steps):
    scheduler = None
    if scheduler_cfg is not None:
        assert scheduler_cfg.get("name") is not None, "Scheduler name must be provided in the config"
        scheduler_cfg = scheduler_cfg.copy()
        scheduler_name = scheduler_cfg["name"]
        del scheduler_cfg["name"]

        
        if scheduler_cfg.get("num_warmup_steps") is not None:
            num_warmup_steps = scheduler_cfg["num_warmup_steps"]
            del scheduler_cfg["num_warmup_steps"]
        elif scheduler_cfg.get("warmup_ratio") is not None:
            num_warmup_steps = int(scheduler_cfg["warmup_ratio"] * total_training_steps)
            del scheduler_cfg["warmup_ratio"]
        else:
            # Default warmup ratio of 5%
            num_warmup_steps = int(total_training_steps * 0.05)
        
        scheduler = get_scheduler_fn(
            scheduler_name,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            **scheduler_cfg
        )
        
    return scheduler

def get_scheduler_step_type(scheduler):
    """
    Returns a string describing when to call scheduler.step():
    - "batch": call after each training batch
    - "epoch": call after each training epoch
    - "metric": call after each epoch, passing a validation metric
    """
    is_hf = hasattr(scheduler, "optimizer") and hasattr(scheduler, "last_epoch") and \
            not isinstance(scheduler, (
                StepLR, MultiStepLR, ExponentialLR,
                ReduceLROnPlateau, OneCycleLR, CyclicLR, LambdaLR
            ))

    if is_hf or isinstance(scheduler, (OneCycleLR, CyclicLR)):
        return "batch"
    elif isinstance(scheduler, ReduceLROnPlateau):
        if scheduler.mode == "min":
            return "metric_min"
        elif scheduler.mode == "max":
            return "metric_max"
    elif scheduler is not None:
        return "epoch"

    return None

def create_optimizer(opt_name, model, lr, decay):
    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise NotImplementedError(f"Optimizer {opt_name} is not implemented")
    return optimizer

def floats_to_filename(data, max_items=5):
    def format_num(x):
        return f"{x:.2f}"  # keep dot, round to 2 decimals
    
    if isinstance(data, float):
        filename = format_num(data)
    
    elif isinstance(data, np.ndarray):
        flat = data.flatten()
        items = [format_num(x) for x in flat[:max_items]]
        filename = "_".join(items)
        if flat.size > max_items:
            filename += f"_len{flat.size}"
    
    elif isinstance(data, torch.Tensor):
        flat = data.flatten().tolist()
        items = [format_num(x) for x in flat[:max_items]]
        filename = "_".join(items)
        if len(flat) > max_items:
            filename += f"_len{len(flat)}"
    
    else:
        raise TypeError("Unsupported type")
    
    return filename

def log_config(writer: SummaryWriter, cfg, tag: str = "config"):
    cfg_json = cfg.dumps(sort_keys=False, default_flow_style=False)
    writer.add_text(tag, f"```yaml\n{cfg_json}\n```")

def log_model_arch(writer: SummaryWriter, model_arch, tag: str = "arch"):
    cfg_json = json.dumps(model_arch, indent=2)
    writer.add_text(tag, f"```json\n{cfg_json}\n```")

def get_current_lr(optimizer, scheduler):
    """Get the current learning rate from optimizer/scheduler."""
    lr = None
    if scheduler:
        lr = scheduler.get_last_lr()[0]
    else:
        lr = optimizer.param_groups[0]["lr"]
    return lr


def get_single_accuracy_metric(acc):
    """Extract a single accuracy value from various accuracy formats."""
    if isinstance(acc, np.ndarray) and len(acc) > 0:
        return acc[0]
    if isinstance(acc, torch.Tensor) and acc.numel() > 0 and acc.ndim != 0:
        return acc[0]
    return acc


def restore_training_state(cfg, prev_cfg, save_file, model, optimizer, scheduler):
    """Restore training state from checkpoint including model weights, optimizer, scheduler, and RNG states."""
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

        if prev_cfg is not None:
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

        # Restore RNG states
        torch.set_rng_state(saved_state["rng_state"]["torch"].clone().type(torch.ByteTensor))
        torch.cuda.set_rng_state_all([s.clone().type(torch.ByteTensor) for s in saved_state["rng_state"]["cuda"]])
        if "numpy_seed" in saved_state["rng_state"]:
            np.random.seed(saved_state["rng_state"]["numpy_seed"])
            random.seed(saved_state["rng_state"]["python_seed"])
        else:
            np.random.set_state(saved_state["rng_state"]["numpy"])
            random.setstate(saved_state["rng_state"]["python"])
    
    return valid_acc, start_epoch


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, valid_acc, save_path):
    """Save training checkpoint including model, optimizer, scheduler states and RNG states."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_acc": valid_acc,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy_seed": np.random.get_state()[1][0],  # pyright: ignore[reportArgumentType]
                "python_seed": random.getstate()[1][0],
            }
        },
        save_path
    )

def save_best_model_checkpoint(model_name, model, optimizer, scheduler, scaler, best_valid_acc, run_dir, i, mean_val_acc):
    save_file_name = None
    if best_valid_acc is None or \
            (get_single_accuracy_metric(mean_val_acc) >
              get_single_accuracy_metric(best_valid_acc)):
        print(f"Improved validation accuracy (prev: {best_valid_acc}, curr: {mean_val_acc})! Saving ...", end="\t", flush=True)
        best_valid_acc = mean_val_acc
        save_file_name = f"{model_name}_{i:07d}_{floats_to_filename(best_valid_acc)}.pt"
    elif i % 100 == 0:
        print(f"Saving for epoch checkpointing (acc: {mean_val_acc})...", end="\t", flush=True)
        save_file_name = f"{i:07d}_{model_name}.pt"

    if save_file_name is not None:
        save_checkpoint(model, optimizer, scheduler, scaler, i, mean_val_acc, run_dir / save_file_name)
        print("Done", flush=True)

    return best_valid_acc


def setup_model_directories(model_name, train_from_checkpoint, configs_dir, runs_dir):
    """Setup model directories and find checkpoint files if needed.
    
    Returns:
        tuple: (save_file, prev_run_dir, run_dir, model_arch_file)
    """
    run_dir = runs_dir / get_log_dir_name(model_name)
    save_file, prev_run_dir = None, None
    
    if isinstance(train_from_checkpoint, bool) and train_from_checkpoint == True:
        prev_run_dir = find_latest_run_dir(model_name)
    elif isinstance(train_from_checkpoint, str):
        save_file = Path(train_from_checkpoint).absolute()
        assert save_file.exists() and save_file.is_file(), f"Checkpoint file does not exist or is not a file:\n\t{save_file}\n  Please give the right checkpoint file name in train_from_checkpoint"
        prev_run_dir = save_file.parent
        assert prev_run_dir.exists() and prev_run_dir.is_dir(), f"Run directory does not exist:\n\t{prev_run_dir}\n\tPlease give the right checkpoint directory name in train_from_checkpoint"

    if prev_run_dir is not None: 
        # Found existing run directory
        assert prev_run_dir.is_dir(), f"Expected run_dir ({prev_run_dir}) to be a folder"

        model_arch_file = prev_run_dir / f"{model_name}.json"
        print(f"Found existing run directory:\n\t{prev_run_dir}\n\twith model architecture file {model_arch_file.name}")
        if save_file is None:
            save_file = find_latest_wgts(prev_run_dir, model_name)
            assert save_file is not None and save_file.exists(), f"Weight file not found in prev run dir:\n\t{prev_run_dir}"
        
        copy_file_to_dir(model_arch_file, run_dir)
    else:
        # New training run
        model_arch_file = configs_dir / f"{model_name}.json"
        print(f"Creating new run directory {run_dir} with model architecture file {model_arch_file.name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        copy_file_to_dir(model_arch_file, run_dir)

    return save_file, prev_run_dir, run_dir, model_arch_file

def load_training_modules(
        create_model_fn, 
        model_name: str, 
        data_shapes: dict, 
        cfg, 
        load_preprocessor: Optional[Callable[[dict, dict, Path, str, SummaryWriter], tuple]] = None):
    train_from_checkpoint = cfg.train_from_checkpoint
    run_dir = runs_dir / get_log_dir_name(model_name)

    save_file, prev_run_dir, prev_cfg = None, None, None
    if isinstance(train_from_checkpoint, bool) and train_from_checkpoint == True:
        prev_run_dir = find_latest_run_dir(model_name)
    elif isinstance(train_from_checkpoint, str):
        save_file = Path(train_from_checkpoint).absolute()
        assert save_file.exists() and save_file.is_file(), f"Checkpoint file does not exist or is not a file:\n\t{save_file}\n  Please give the right checkpoint file name in train_from_checkpoint"
        prev_run_dir = save_file.parent
        assert prev_run_dir.exists() and prev_run_dir.is_dir(), f"Run directory does not exist:\n\t{prev_run_dir}\n\tPlease give the right checkpoint directory name in train_from_checkpoint"

    if prev_run_dir is not None: 
        # Found existing run directory
        assert prev_run_dir.is_dir(), f"Expected run_dir ({prev_run_dir}) to be a folder"

        model_arch_file = prev_run_dir / f"{model_name}.json"
        print(f"Found existing run directory:\n\t{prev_run_dir}\n\twith model architecture file {model_arch_file.name}")
        if save_file is None:
            save_file = find_latest_file(prev_run_dir, f"*_{model_name}.pt")
            assert save_file is not None and save_file.exists(), f"Weight file not found in prev run dir:\n\t{prev_run_dir}"
        
        copy_file_to_dir(model_arch_file, run_dir)

        prev_config_file = find_latest_file(prev_run_dir, "config*.yaml")
        assert prev_config_file is not None, f"Config file not found in prev run dir:\n\t{prev_run_dir}"

        prev_cfg = load_config(prev_config_file, train=True)
        assert isinstance(prev_cfg, TrainConfig), f"Previous config is not a TrainConfig: {prev_config_file}"

        prev_cfg.save(run_dir / f"prev-config_{prev_run_dir.name}.yaml")
    else:
        # New training run
        model_arch_file: Path = configs_dir / f"{model_name}.json"

        print(f"Creating new run directory {run_dir} with model architecture file {model_arch_file.name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        copy_file_to_dir(model_arch_file, run_dir)


    writer = SummaryWriter(log_dir=run_dir)

    with open(model_arch_file) as f:
        model_config : dict = json.load(f)

    log_model_arch(writer, model_config)

    device : str = cfg.device
    preprocessor = None
    if load_preprocessor is not None:
        data_shapes, preprocessor = load_preprocessor(
            model_config, data_shapes, model_arch_file, device, writer)

    model = create_model_fn(model_config, data_shapes, extra_config=cfg.extra_config)

    optimizer = create_optimizer(cfg.optimizer, model, cfg.learning_rate, cfg.weight_decay)

    total_training_steps = cfg.epochs * ((cfg.train_dataset_size - 1) // cfg.batch_size + 1)
    scheduler = create_scheduler(cfg.scheduler, optimizer, total_training_steps)
    scaler = None

    valid_acc, start_epoch = restore_training_state(
        cfg, prev_cfg, save_file, model, optimizer, scheduler)

    model.to(device)

    cfg.save(run_dir / f"config.yaml")
    log_config(writer, cfg)

    return model, optimizer, scheduler, scaler, start_epoch, valid_acc, run_dir, writer, preprocessor

def train_one_epoch(single_step_fn, train_dataloader, model, optimizer, loss_fn, acc_fn, scheduler,
                    scheduler_step_type, epoch, writer, debug):
    index_train = 0
    tot_train_loss = 0
    tot_train_acc  = 0

    model.train()
    total_steps = len(train_dataloader)
    for step, data in enumerate(tqdm(train_dataloader, desc=f"Train {epoch}")):
        torch.cuda.empty_cache()
        optimizer.zero_grad()


        batch = data[0] if isinstance(data, list) else data
        if debug:
            size_accum = 0
            for file in batch.file_path:
                file_path = Path(file)
                size_accum += file_path.stat().st_size
            print(f"{sizeof_fmt(size_accum)}", flush=True)

        try:
            loss, acc = single_step_fn(model, data, loss_fn, acc_fn, epoch, step)
        except torch.OutOfMemoryError as e:
            total = 0
            for file in batch.file_path:
                file_path = Path(file)
                size = file_path.stat().st_size
                total += size
                print(f"{file}, {sizeof_fmt(size)}", flush=True)
            print(f"Total batch size: {sizeof_fmt(total)}", flush=True)
            raise e
        except Exception as e:
            print(f"Files: {chr(10).join(batch.file_path)}")
            raise e

        loss.backward(retain_graph=False)
        optimizer.step()
        if scheduler_step_type == "batch":
            scheduler.step()
        
        batch_size = batch.batch_size
        tot_train_loss += loss.detach().cpu().item() * batch_size
        tot_train_acc  += acc * batch_size
        index_train    += batch_size

        if debug:
            print_gpu_mem(f"Train, Step: {step}, Epoch {epoch}")

            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param.data, epoch * total_steps + step)
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad, epoch * total_steps + step)
                    # Print ratio of how many gradients are zero
                    print(f"Step {step}, Param {name}, Grad Non-Zero Ratio: {torch.count_nonzero(param.grad) / param.grad.numel()}")

        step += 1

    mean_train_loss = tot_train_loss / index_train
    mean_train_acc  = tot_train_acc / index_train

    if scheduler_step_type == "epoch":
            scheduler.step()
    
    return mean_train_loss, mean_train_acc

def validate_model(single_step_fn, val_dataloader, model, loss_fn, acc_fn, epoch):
    index_val = 0
    tot_val_loss = 0
    tot_val_acc = 0

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(val_dataloader, desc=f"Valid {epoch}")):
            batch = data[0] if isinstance(data, list) else data
            loss, acc = single_step_fn(model, data, loss_fn, acc_fn, epoch, step)
            tot_val_loss += loss.item() * batch.batch_size
            tot_val_acc  += acc * batch.batch_size
            index_val    += batch.batch_size

    mean_val_loss   = tot_val_loss / index_val
    mean_val_acc    = tot_val_acc / index_val
    return mean_val_loss, mean_val_acc


def _default_arg_defs():
    """Base arguments common to all training entrypoints."""
    return {
        "--config": dict(type=str, required=True),
        "--debug": dict(action="store_true"),
        "--epochs": dict(type=int, default=None),
        "--learning_rate": dict(type=float, default=None),
        "--weight_decay": dict(type=float, default=None),
        "--train_from_checkpoint": dict(type=str, default=None),
        "--optimizer": dict(type=str, default=None),
        "--scheduler": dict(type=str, default=None),
    }

def _default_handlers():
    """Base handlers for common args."""

    def handle_scheduler(val, args, cfg):
        args.__dict__["scheduler"] = json.loads(val)
        print(f"Overriding config value scheduler with {val}")
        cfg.__dict__["scheduler"] = args.scheduler

    def handle_checkpoint(val, args, cfg):
        if val.lower() == "true":
            args.__dict__["train_from_checkpoint"] = True
        elif val.lower() == "false":
            args.__dict__["train_from_checkpoint"] = False
        print(f"Overriding config value train_from_checkpoint with {args.train_from_checkpoint}")
        cfg.__dict__["train_from_checkpoint"] = args.train_from_checkpoint

    return {
        "scheduler": handle_scheduler,
        "train_from_checkpoint": handle_checkpoint,
    }

def parse_and_run(
    config_class,
    main_fn,
    arg_definitions={},
    special_handlers={},
):
    """
    Shared entrypoint runner with defaults + overrides.

    Args:
        config_class (type): Expected config class (e.g., TrainConfig).
        main_fn (callable): Main training function.
        configs_dir (Path): Directory where YAML configs are stored.
        arg_definitions (dict): Extra or overriding argparse definitions.
        special_handlers (dict): Extra or overriding handlers for arguments.
    """

    # merge arg defs (default + custom)
    merged_arg_defs = _default_arg_defs()
    merged_arg_defs.update(arg_definitions)

    parser = argparse.ArgumentParser()
    for name, kwargs in merged_arg_defs.items():
        parser.add_argument(name, **kwargs)
    args = parser.parse_args()

    # load config
    config_path = configs_dir / f"{args.config}.yaml"
    print(f"Loading config from {config_path}")
    assert config_path.exists(), f"Config file {config_path} does not exist."

    cfg = load_config(config_path, train=True)

    # merge handlers (default + custom)
    merged_handlers = _default_handlers()
    merged_handlers.update(special_handlers)

    # apply overrides
    for arg_k, arg_v in vars(args).items():
        if arg_k not in ["config", "debug"] and arg_v is not None:
            if arg_k in merged_handlers:
                merged_handlers[arg_k](arg_v, args, cfg)
            else:
                print(f"Overriding config value {arg_k} with {arg_v}")
                cfg.__dict__[arg_k] = arg_v

    debug = args.debug

    assert isinstance(cfg, config_class), f"Config loaded is not {config_class}, got {type(cfg)}"

    make_deterministic(cfg.seed)
    main_fn(cfg, debug)
    print("Training Done!")


def setup_training(
    cfg,
    create_model_fn,
    data_loader_fn,
    load_preprocessor_fn=None
):
    """
    Sets up dataloaders, model, optimizer, scheduler, scaler, writer, etc.

    Returns:
        (train_dataloader, val_dataloader, model, optimizer, scheduler,
         scaler, start_epoch, best_valid_acc, run_dir, writer, preprocessor)
    """
    device, model_name, dataset, batch_size = (
        cfg.device, cfg.model_name, cfg.dataset, cfg.batch_size
    )
    epochs, lr, decay = cfg.epochs, cfg.learning_rate, cfg.weight_decay

    print(f"Training {model_name} on {dataset}")
    print(f"  epochs: {epochs}\n  batch size: {batch_size}\n  learning rate: {lr}\n  weight decay: {decay}")

    # --- Load data
    train_dataloader, val_dataloader, _ = data_loader_fn(dataset, device, batch_size, cfg)
    sample = next(iter(train_dataloader))
    data_sample = sample[0] if isinstance(sample, (tuple, list)) else sample
    data_shapes = get_data_shape(data_sample)

    # --- Load model, optimizer, scheduler, scaler
    model, optimizer, scheduler, scaler, start_epoch, best_valid_acc, run_dir, writer, preprocessor = \
        load_training_modules(create_model_fn, model_name, data_shapes, cfg, load_preprocessor=load_preprocessor_fn)

    if best_valid_acc is not None:
        print(f"  Best Valid Acc: {best_valid_acc}")

    return (train_dataloader, val_dataloader, model, optimizer, scheduler,
            scaler, start_epoch, best_valid_acc, run_dir, writer, preprocessor)

def training_loop(
    cfg,
    debug,
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    scheduler,
    scaler,
    start_epoch,
    best_valid_acc,
    run_dir,
    writer,
    loss_fn,
    acc_fn,
    single_step_fn,
    log_training_metrics
):
    """
    Standard training loop with validation, checkpointing, and logging.
    """
    scheduler_step_type = get_scheduler_step_type(scheduler)
    epochs = cfg.epochs

    for i in range(start_epoch, epochs):
        gc.collect()
        torch.cuda.empty_cache()

        mean_train_loss, mean_train_acc = train_one_epoch(
            single_step_fn, train_dataloader, model, optimizer,
            loss_fn, acc_fn, scheduler, scheduler_step_type,
            i, writer, debug)

        mean_val_loss, mean_val_acc = validate_model(
            single_step_fn, val_dataloader, model, loss_fn, acc_fn, i)

        if scheduler_step_type == "metric_min":
            scheduler.step(mean_val_loss)  # pyright: ignore
        elif scheduler_step_type == "metric_max":
            scheduler.step(mean_val_acc)   # pyright: ignore

        best_valid_acc = save_best_model_checkpoint(
            cfg.model_name, model, optimizer,
            scheduler, scaler, best_valid_acc,
            run_dir, i, mean_val_acc)

        log_training_metrics(
            writer, i, mean_train_loss, mean_train_acc,
            mean_val_loss, mean_val_acc,
            get_current_lr(optimizer, scheduler)
        )

    writer.close()
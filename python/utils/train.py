from transformers import get_scheduler as hf_get_scheduler
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau,
    OneCycleLR, CyclicLR, LambdaLR
)

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
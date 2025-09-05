import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path

from train_vecparams_e2e import load_vecparam_data, get_best_vf_if_loss_acc, get_weighted_loss_acc
from utils.config import TrainConfig

from utils.infer import load_grami_enc, setup_testing
from utils.train import (
    parse_and_run,
    setup_training,
    training_loop,
)

from models.VecParams.model import VecParamsClassifier

from torch.utils.tensorboard import SummaryWriter


def main(cfg, debug, train):
    create_model_fn = lambda model_config, data_shapes, extra_config: \
        VecParamsClassifier(model_config["classifier"], data_shapes, extra_config=extra_config)

    # Loss + acc
    loss_fn = nn.BCELoss()
    acc_fn = lambda outputs, one_hot: (outputs.argmax(-1) == one_hot.argmax(-1)).float()

    vf_classes = torch.tensor([1, 2, 4, 8, 16, 32, 64], device=cfg.device, dtype=torch.long)
    if_classes = torch.tensor([1, 2, 4, 8, 16], device=cfg.device, dtype=torch.long)
    classes = torch.cartesian_prod(vf_classes, if_classes)

    if train:
        # Setup
        train_dl, val_dl, model, optimizer, scheduler, scaler, start_epoch, \
            best_acc, run_dir, writer, GraMI_enc = \
            setup_training(
                cfg,
                create_model_fn=create_model_fn,
                data_loader_fn=load_vecparam_data,
                load_preprocessor_fn=load_grami_enc,
            )

        # Single step
        single_step_fn = lambda model, data, loss_fn, acc_fn, epoch, data_index: \
            single_step(model, data[0], data[1], loss_fn, acc_fn, GraMI_enc, classes)

        # Loop
        training_loop(cfg, debug, train_dl, val_dl, model, optimizer, 
                    scheduler, scaler, start_epoch, best_acc, run_dir, 
                    writer, loss_fn, acc_fn, single_step_fn, log_training_metrics)
    else:
        tot_val_acc, tot_val_loss, index_val = 0, 0, 0

        test_dataloader, model, run_dir, csv_file, GraMI_enc = \
            setup_testing(cfg, create_model_fn, load_vecparam_data, load_grami_enc)
        
        df = pd.DataFrame(columns=["file_name", "loss", "acc", "pred", "target"])
        assert GraMI_enc is not None
        
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                batch, labels = data

                z_A, z_V = GraMI_enc(batch)

                assert isinstance(model, VecParamsClassifier)
                batch_size = len(labels)
                logits = model.forward(
                    z_A.detach(), {k: v.detach() for k, v in z_V.items()},
                    {k: batch[k].batch for k in z_V.keys()}, batch_size
                )

                targets = (labels[:, None, :] == classes[None, :, :]).all(dim=-1).float()

                for file_name, prob, target in zip(batch.file_path, logits, targets):
                    loss = loss_fn(prob, target.to(torch.float32)).item()
                    acc = acc_fn(prob, target).item()

                    new_row = pd.Series({
                        "file_name": Path(file_name).name, 
                        "loss": loss, 
                        "acc": acc,
                        "pred": tuple(classes[prob.argmax().item()].tolist()),
                        "target": tuple(classes[target.argmax().item()].tolist())
                    })
                    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

                    tot_val_loss += loss
                    tot_val_acc  += acc
                    index_val    += 1

                df.to_csv(csv_file, index=False)

            test_loss = tot_val_loss / index_val if index_val > 0 else 0
            test_acc = tot_val_acc / index_val if index_val > 0 else 0  

            print(f"\n\tTest | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}\n")
            with open(csv_file.with_suffix(".txt"), "w") as f:
                f.write(f"Test | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}\n")

def single_step(model, batch, labels, loss_fn, acc_fn, GraMI_enc, classes):
    z_A, z_V = GraMI_enc(batch)
    assert isinstance(model, VecParamsClassifier)
    batch_size = len(batch[list(batch.x_dict.keys())[0]].ptr) - 1
    assert all(len(batch[k].ptr) == batch_size + 1 for k in batch.x_dict.keys()), "Pointer lengths do not match"

    logits = model.forward(
        z_A.detach(), {k: v.detach() for k, v in z_V.items()},
        {k: batch[k].batch for k in z_V.keys()}, batch_size
    )

    loss, acc = get_best_vf_if_loss_acc(labels, loss_fn, acc_fn, classes, logits)
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
        main_fn=main,
    )
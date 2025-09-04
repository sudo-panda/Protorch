import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

from utils.config import TrainConfig

from utils.dataset import VecParamsDataset
from utils.infer import setup_testing
from utils.paths import get_data_paths
from utils.train import (
    parse_and_run,
    setup_training,
    training_loop,
)

from models.VecParams.model import VecParamsE2EModel

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def load_vecparam_data(dataset, device, batch_size, cfg):
    ################ Load data ################
    _, data_path, csv_file = get_data_paths(dataset)

    def df_to_inputs(df: pd.DataFrame, path: Path):
        df["file_path"] = df["pt_file"].apply(lambda x: str(path / x))
        input_list = df.reset_index().to_dict('records')
        return input_list

    if isinstance(csv_file, dict):
        if all(key in csv_file for key in ["train", "val", "test"]):
            train_df = pd.read_csv(csv_file["train"])
            val_df = pd.read_csv(csv_file["val"])
            test_df = pd.read_csv(csv_file["test"])

            train_data_path = data_path["train"]
            val_data_path = data_path["val"]
            test_data_path = data_path["test"]
        else:
            train_val_df = pd.read_csv(csv_file["train"])
            test_df = pd.read_csv(csv_file["test"])
            train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=cfg.seed)

            train_data_path = data_path["train"]
            val_data_path = data_path["train"]
            test_data_path = data_path["test"]
    else:
        df = pd.read_csv(csv_file)
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=cfg.seed)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=cfg.seed)

        train_data_path = data_path
        val_data_path = data_path
        test_data_path = data_path
    
    # train_files, val_files, test_files = file_list[0:2], file_list[2:3], file_list[3:4]
    # train_devmap, val_devmap, test_devmap = devmap_list[0:2], devmap_list[2:3], devmap_list[3:4]

    train_inputs = df_to_inputs(train_df, train_data_path)
    val_inputs   = df_to_inputs(val_df,   val_data_path)
    test_inputs  = df_to_inputs(test_df,  test_data_path)

    cfg["train_dataset_size"] = len(train_inputs)
    cfg["val_dataset_size"] = len(val_inputs)
    cfg["test_dataset_size"] = len(test_inputs)

    train_dataloader = DataLoader(VecParamsDataset(train_inputs, device=device), batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(VecParamsDataset(val_inputs,   device=device), batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(VecParamsDataset(test_inputs,  device=device), batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def main(cfg, debug, train):
    # Loss + acc
    loss_fn = nn.BCELoss()
    acc_fn = lambda outputs, one_hot: (outputs.argmax(-1) == one_hot.argmax(-1)).float()

    classes = {
        "vf_classes": torch.tensor([1, 2, 4, 8, 16, 32, 64], device=cfg.device, dtype=torch.long),
        "if_classes": torch.tensor([1, 2, 4, 8, 16], device=cfg.device, dtype=torch.long),
    }

    if train:
        # Setup
        train_dl, val_dl, model, optimizer, scheduler, scaler, start_epoch, \
            best_acc, run_dir, writer, _ = \
            setup_training(
                cfg,
                create_model_fn=VecParamsE2EModel,
                data_loader_fn=load_vecparam_data,
            )


        # Single step
        single_step_fn = lambda model, data, loss_fn, acc_fn, epoch, data_index: \
            single_step(model, data[0], data[1], loss_fn, acc_fn, classes)

        # Loop
        training_loop(cfg, debug, train_dl, val_dl, model, optimizer, 
                    scheduler, scaler, start_epoch, best_acc, run_dir, 
                    writer, loss_fn, acc_fn, single_step_fn, log_training_metrics)
    else:
        tot_val_acc, tot_val_loss, index_val = 0, 0, 0

        test_dataloader, model, run_dir, csv_file, _ = \
            setup_testing(cfg, VecParamsE2EModel, load_vecparam_data)
        
        df = pd.DataFrame(columns=["file_name", "loss", "acc", "pred", "target"])
        
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                batch, labels = data

                assert isinstance(model, VecParamsE2EModel)
                vf_logits, if_logits = model.forward(batch)

                
                labels_to_onehot_vf = (labels[:, 0].unsqueeze(1) == classes["vf_classes"]).float()
                labels_to_onehot_if = (labels[:, 1].unsqueeze(1) == classes["if_classes"]).float()

                for file_name, vf_prob, if_prob, vf_labels, if_labels in zip(batch.file_path, vf_logits, if_logits, labels_to_onehot_vf, labels_to_onehot_if):
                    vf_loss = loss_fn(vf_prob, vf_labels.to(torch.float32)).item()
                    if_loss = loss_fn(if_prob, if_labels.to(torch.float32)).item()
                    loss = vf_loss + if_loss

                    vf_acc = acc_fn(vf_prob, vf_labels).item()
                    if_acc = acc_fn(if_prob, if_labels).item()
                    acc = (vf_acc + if_acc) / 2

                    new_row = pd.Series({
                        "file_name": Path(file_name).name, 
                        "loss": loss, 
                        "acc": acc,
                        "pred": (classes['vf_classes'][vf_prob.argmax().item()].item(), classes['if_classes'][if_prob.argmax().item()].item()),
                        "target": (classes['vf_classes'][vf_labels.argmax().item()].item(), classes['if_classes'][if_labels.argmax().item()].item())
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

def single_step(model, batch, labels, loss_fn, acc_fn, classes):
    assert isinstance(model, VecParamsE2EModel)
    logits = model(batch)
    loss, acc = get_weighted_loss_acc(labels, loss_fn, acc_fn, classes, logits)
    return loss, acc

def get_best_vf_if_loss_acc(labels, loss_fn, acc_fn, classes, logits):
    vf_logits, if_logits = logits
    labels_to_onehot_vf = (labels[:, 0].unsqueeze(1) == classes["vf_classes"]).float()
    labels_to_onehot_if = (labels[:, 1].unsqueeze(1) == classes["if_classes"]).float()

    loss = loss_fn(vf_logits, labels_to_onehot_vf).mean() + loss_fn(if_logits, labels_to_onehot_if).mean()
    acc = acc_fn(vf_logits, labels_to_onehot_vf).mean() + acc_fn(if_logits, labels_to_onehot_if).mean()
    return loss, acc

def get_weighted_loss_acc(labels, loss_fn, acc_fn, classes, logits):
    def top1_utility_ratio_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = logits.argmax(dim=1)
        util_pred = targets.gather(1, pred.unsqueeze(1)).squeeze(1)
        util_best = targets.max(dim=1).values
        return (util_pred / (util_best + 1e-12)).mean()

    loss = F.kl_div(
        F.log_softmax(logits, dim=1),
        labels,
        reduction='batchmean'
    )
    acc = top1_utility_ratio_from_logits(logits, labels)
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
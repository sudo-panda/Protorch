from pathlib import Path
from typing import Union
import torch
import torch.nn as nn

from utils.infer import setup_testing
from utils.paths import get_data_paths
from utils.config import TestConfig, TrainConfig
from utils.train import (
    parse_and_run,
    setup_training,
    training_loop,
)

from utils.dataset import DevmapDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from models.Devmap import DevmapE2EModel


def load_devmap_data(dataset: str, device: str, batch_size: int, cfg: Union[TestConfig, TrainConfig]):
    ################ Load data ################
    _, data_path, csv_file = get_data_paths(dataset)

    assert isinstance(csv_file, Path), f"Not Implemented support for {type(csv_file)} as CSV file var"
    with open(csv_file) as f:
        df = pd.read_csv(f)

    df["file_path"] = df["pt_file"].apply(lambda x: str(data_path / x))
    input_list = df.reset_index()[["file_path", "comp", "rational", "mem", "localmem", "coalesced", "atomic", "transfer", "wgsize"]].to_dict('records')

    devmap_list = df["device"].tolist()
    assert len(input_list) == len(devmap_list), "File list and device list must have the same length"

    train_inputs, temp_inputs, train_devmap, temp_devmap = train_test_split(input_list,  devmap_list, test_size=0.4, random_state=cfg.seed)
    val_inputs,   test_inputs, val_devmap,   test_devmap = train_test_split(temp_inputs, temp_devmap, test_size=0.5, random_state=cfg.seed)
    # train_files, val_files, test_files = file_list[0:2], file_list[2:3], file_list[3:4]
    # train_devmap, val_devmap, test_devmap = devmap_list[0:2], devmap_list[2:3], devmap_list[3:4]

    cfg["train_dataset_size"] = len(train_inputs)
    cfg["val_dataset_size"] = len(val_inputs)
    cfg["test_dataset_size"] = len(test_inputs)

    train_dataloader = DataLoader(DevmapDataset(train_inputs, train_devmap, device=device), batch_size=batch_size, shuffle=True)
    assert isinstance(train_dataloader.dataset, DevmapDataset), "Expected train_dataloader.dataset to be an instance of DevmapDataset"
    mean_std_dict = train_dataloader.dataset.mean_std_dict
    val_dataloader   = DataLoader(DevmapDataset(val_inputs,   val_devmap,   device=device, mean_std_dict=mean_std_dict), batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(DevmapDataset(test_inputs,  test_devmap,  device=device, mean_std_dict=mean_std_dict), batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def main(cfg, debug, train=True):

    loss_fn = nn.BCELoss()
    acc_fn = lambda preds, labels: ((preds >= 0.5).to(torch.int32) == labels).type(torch.float32)

    if train:
        # Setup
        train_dl, val_dl, model, optimizer, scheduler, scaler, \
            start_epoch, best_acc, run_dir, writer, _ = \
            setup_training(
                cfg,
                create_model_fn=DevmapE2EModel,
                data_loader_fn=load_devmap_data,
            )
        

        single_step_fn = lambda model, data, loss_fn, acc_fn, epoch, data_index: \
            single_step(model, data[0], data[1], loss_fn, acc_fn)

        training_loop(cfg, debug, train_dl, val_dl, model, 
                    optimizer, scheduler, scaler,
                    start_epoch, best_acc, run_dir, writer, 
                    loss_fn, acc_fn, single_step_fn, log_training_metrics)
    else:
        tot_val_acc, tot_val_loss, index_val = 0, 0, 0

        test_dataloader, model, run_dir, csv_file, _ = \
            setup_testing(cfg, DevmapE2EModel, load_devmap_data)
        
        df = pd.DataFrame(columns=["file_name", "loss", "acc", "pred", "target"])
        
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                batch, labels = data

                assert isinstance(model, DevmapE2EModel)
                logits = model.forward(batch)

                probs = torch.sigmoid(logits)

                for file_name, prob, label in zip(batch.file_path, probs, labels):
                    loss = loss_fn(prob, label.to(torch.float32)).item()
                    acc = acc_fn(prob, label).item()
                    new_row = pd.Series({
                        "file_name": Path(file_name).name, 
                        "loss": loss, 
                        "acc": acc,
                        "pred": "GPU" if (prob > 0.5) else "CPU",
                        "target": "GPU" if (label == 1) else "CPU"
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


def single_step(model, batch, labels, loss_fn, acc_fn):
    logits = model(batch)
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
        main_fn=main,
    )
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from utils.infer import load_grami_enc, setup_testing
from utils.train import (
    parse_and_run,
    setup_training,
    training_loop,
)

from train_devmap import load_devmap_data

from models.Devmap.model import DevmapClassifier

from torch.utils.tensorboard import SummaryWriter

def main(cfg, debug, train=True):
    create_model_fn = lambda model_config, data_shapes, extra_config: \
        DevmapClassifier(model_config["classifier"], data_shapes, extra_config=extra_config)


    # Loss + acc
    loss_fn = nn.BCELoss()
    acc_fn = lambda preds, labels: ((preds >= 0.5).to(torch.int32) == labels).type(torch.float32)
    if train:
        # Setup
        train_dl, val_dl, model, optimizer, scheduler, scaler, start_epoch, \
            best_acc, run_dir, writer, GraMI_enc = \
            setup_training(
                cfg,
                create_model_fn=create_model_fn,
                data_loader_fn=load_devmap_data,
                load_preprocessor_fn=load_grami_enc,
            )

        # Single step
        single_step_fn = lambda model, data, loss_fn, acc_fn, epoch, data_index: \
            single_step(model, data[0], data[1], loss_fn, acc_fn, GraMI_enc)

        # Loop
        training_loop(cfg, debug, train_dl, val_dl, model, optimizer, 
                    scheduler, scaler, start_epoch, best_acc, run_dir, 
                    writer, loss_fn, acc_fn, single_step_fn, log_training_metrics)
    else:
        tot_val_acc, tot_val_loss, index_val = 0, 0, 0

        test_dataloader, model, run_dir, csv_file, GraMI_enc = \
            setup_testing(cfg, create_model_fn, load_devmap_data, load_grami_enc)
        
        df = pd.DataFrame(columns=["file_name", "loss", "acc", "pred", "target"])
        assert GraMI_enc is not None
        
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                batch, labels = data

                z_A, z_V = GraMI_enc(batch)

                assert isinstance(model, DevmapClassifier)
                batch_size = len(labels)
                logits = model(
                    z_A.detach(), {k: v.detach() for k, v in z_V.items()}, 
                    batch.comp, batch.mem, batch.localmem, 
                    batch.coalesced, batch.transfer, batch.wgsize, 
                    {k: batch[k].batch for k in z_V.keys()}, batch_size
                )

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

def single_step(model, batch, labels, loss_fn, acc_fn, GraMI_enc):
    z_A, z_V = GraMI_enc(batch)
    assert isinstance(model, DevmapClassifier)
    batch_size = len(labels)
    logits = model(
        z_A.detach(), {k: v.detach() for k, v in z_V.items()}, 
        batch.comp, batch.mem, batch.localmem, 
        batch.coalesced, batch.transfer, batch.wgsize, 
        {k: batch[k].batch for k in z_V.keys()}, batch_size
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
        main_fn=main,
    )
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from utils.common import get_data_shape, get_log_dir_name 
from utils.config import epochs, train_from_checkpoint, lr, decay, batch_size, world_size
from dataset import HecBenchDataset
from train import train_files, val_files, single_step
from models.GraMI import GraMI

from paths import GraMI_path, top_level_path

def task(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}")

    train_dataset = HecBenchDataset(train_files, device=device)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_dataset = HecBenchDataset(val_files, device=device)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    data_sample = next(iter(train_dataloader))

    with open(GraMI_path / "config.json") as f:
        model_config = json.load(f)

    model = GraMI(model_config, get_data_shape(data_sample), device, batch_size)
    if train_from_checkpoint and (GraMI_path / f"{model_config['model_name']}.pt").exists():
        model.load_state_dict(torch.load(GraMI_path / f"{model_config['model_name']}.pt"), strict=True)
    model.to(device)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr, weight_decay=decay)

    writer = SummaryWriter(log_dir=get_log_dir_name(model_config['model_name'])) if rank == 0 else None

    for i in range(epochs):
        train_sampler.set_epoch(i)
        index_train, tot_train_loss, tot_train_edge_acc, tot_train_r2_attr = 0, 0, 0, 0
        index_val, tot_val_loss, tot_val_edge_acc, tot_val_r2_attr = 0, 0, 0, 0

        ddp_model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss, edge_acc, r2_attr = single_step(batch, ddp_model)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item() * batch.batch_size
            tot_train_edge_acc += edge_acc.item() * batch.batch_size
            tot_train_r2_attr += r2_attr.item() * batch.batch_size
            index_train += batch.batch_size

        ddp_model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                loss, edge_acc, r2_attr = single_step(batch, ddp_model)

                tot_val_loss += loss.item() * batch.batch_size
                tot_val_edge_acc += edge_acc.item() * batch.batch_size
                tot_val_r2_attr += r2_attr.item() * batch.batch_size
                index_val += batch.batch_size

        if rank == 0:
            torch.save(model.state_dict(), GraMI_path / f"{model_config['model_name']}.pt")
            writer.add_scalar("Loss/train", tot_train_loss / index_train, i) # type: ignore
            writer.add_scalar("edge-acc/train", tot_train_edge_acc / index_train, i) # type: ignore
            writer.add_scalar("r2-attr/train", tot_train_r2_attr / index_train, i) # type: ignore
            writer.add_scalar("Loss/val", tot_val_loss / index_val, i) # type: ignore
            writer.add_scalar("edge-acc/val", tot_val_edge_acc / index_val, i) # type: ignore
            writer.add_scalar("r2-attr/val", tot_val_r2_attr / index_val, i) # type: ignore
            writer.flush() # type: ignore
    
    if writer:
        writer.close()
    
    dist.destroy_process_group()


def main():
    mp.spawn(task,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
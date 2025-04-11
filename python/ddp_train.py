import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from utils.common import epochs, train_from_checkpoint, lr, decay, batch_size, world_size
from dataset import FunctionGraphDataset
from train import training_list, validation_list, single_step
from GraMI.metrics import loss_fn, acc_fn
from GraMI.model import GraMIModel

from paths import GraMI_path, top_level_path

def task(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}")

    train_dataset = FunctionGraphDataset(training_list, device=device)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_dataset = FunctionGraphDataset(validation_list, device=device)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    data_sample = next(iter(train_dataloader))
    model = GraMIModel(data_sample, 16, 8)
    if train_from_checkpoint and (GraMI_path / "latest.pt").exists():
        model.load_state_dict(torch.load(GraMI_path / "latest.pt"), strict=True)
    model.to(device)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr, weight_decay=decay)

    writer = SummaryWriter() if rank == 0 else None

    for i in range(epochs):
        train_sampler.set_epoch(i)
        index_train, tot_train_loss, tot_train_acc = 0, 0, 0
        index_val, tot_val_loss, tot_val_acc = 0, 0, 0

        ddp_model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss, acc = single_step(batch, ddp_model)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item() * batch.batch_size
            tot_train_acc += acc.item() * batch.batch_size
            index_train += batch.batch_size

        ddp_model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                loss, acc = single_step(batch, ddp_model)

                tot_val_loss += loss.item() * batch.batch_size
                tot_val_acc += acc.item() * batch.batch_size
                index_val += batch.batch_size

        if rank == 0:
            torch.save(model.state_dict(), GraMI_path / "latest.pt")
            writer.add_scalar("Loss/train", tot_train_loss / index_train, i)
            writer.add_scalar("Acc/train", tot_train_acc / index_train, i)
            writer.add_scalar("Loss/val", tot_val_loss / index_val, i)
            writer.add_scalar("Acc/val", tot_val_acc / index_val, i)
            writer.flush()
    
    if writer:
        writer.close()
    
    dist.destroy_process_group()


def main():
    world_size = 2
    mp.spawn(task,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
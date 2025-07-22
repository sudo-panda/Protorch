import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from utils.common import epochs, train_from_checkpoint, lr, decay, batch_size, world_size
from dataset import FunctionGraphDataset
#from train import training_list, validation_list, single_step
import socket
from pathlib import Path
import sys
from models.GraMI.metrics import loss_fn, acc_fn
from GraMI.model import GraMIModel

from paths import GraMI_path, top_level_path


class System():
    """!
    The class representing the system we run. Currently we assume we always run
    on lassen with 'lrun'. Later we will need to make this a parent class and derived
    systems will map to different systems.

    System is a singleton class. Effectively, we can only do training in one system :)
    consisting of multiple nodes. The first time we create an object we read in LSF/lassen
    environmental variables and we identify who we are (rank, localrank etc)
    in the distributed application.
    """

    _instance = None

    @staticmethod
    def _isParallel():
        """!
        'private' method that returns whether this is a distributed run or not.
        """
        if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
            return False
        return True

    @staticmethod
    def _systemName():
        """!
        'private' method that returns the hostname.
        """

        hostName = socket.gethostname()
        return hostName, ''.join(filter(lambda x: not x.isdigit(),hostName))

    @staticmethod
    def _getRank():
        """!
        'private' method returns the id of this process in respect
        to all processes in the distributed allocation. The id should be
        unique to each process.
        """

        if not System._isParallel():
            return 0

        # This value should be set by the lsf backend on lassen
        if 'OMPI_COMM_WORLD_RANK' not in os.environ:
            raise EnvironmentError('OMPI_COMM_WORLD_RANK is not set')

        return int(os.environ['OMPI_COMM_WORLD_RANK'])

    def _getWorldSize():
        """!
        'private' method returns the total number of processes composing
        the distributed application. The id should be
        unique to each process.
        """

        if not System._isParallel():
            return 1

        if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
            raise EnvironmentError('OMPI_COMM_WORLD_SIZE is not set')

        return int(os.environ['OMPI_COMM_WORLD_SIZE'])

    @staticmethod
    def _getLocalWorldSize():
        """!
        'private' method returns the total number of processes executing
        on the same node.
        """

        if not System._isParallel():
            return 1


        if 'OMPI_COMM_WORLD_LOCAL_SIZE' not in os.environ:
            raise EnvironmentError('OMPI_COMM_WORLD_LOCAL_SIZE is not set')

        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])

    @staticmethod
    def _getLocalRank():
        """!
        'private' method returns the total a unique id for the
        running process within the current node.
        """

        if not System._isParallel():
            return 0

        if 'OMPI_COMM_WORLD_LOCAL_RANK' not in os.environ:
            raise EnvironmentError('OMPI_COMM_WORLD_LOCAL_RANK is not set')

        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    @staticmethod
    def _getIPMaster():
        """!
        'private' the host-address of the master process. This is by no means
        a portable solution. It depends mainly on LSF to figure this out.
        """

        if not System._isParallel():
            return socket.gethostname()

        if 'LSB_DJOB_RANKFILE' not in os.environ:
            raise EnvironmentError('LSB_DJOB_RANKFILE is not set')

        with open(os.environ['LSB_DJOB_RANKFILE'], 'r') as fd:
            lines=fd.readlines()
            Addr=lines[2].strip('\n')

        return f'{Addr}'

    def __init__(self):
        self._SName, self._CName = System._systemName()
        self._Rank = System._getRank()
        self._WorldSize = System._getWorldSize()
        self._LRank = System._getLocalRank()
        self._LocalSize = System._getLocalWorldSize()
        self._IPMAddress = System._getIPMaster()
        self._Port = 12321
        self._Parallel = System._isParallel()
        user=os.environ['USER']
        self._NFS = Path(f'/p/gpfs1/{user}/')

    @property
    def Name(self):
        return self._SName

    @Name.setter
    def Name(self, Name):
        self._SName = Name

    @property
    def ClusterName(self):
        return self._CName

    @ClusterName.setter
    def ClusterName(self, ClusterName):
        self._CName = ClusterName

    @property
    def Rank(self):
        return self._Rank

    @Rank.setter
    def Rank(self, Rank):
        self._Rank = Rank

    @property
    def WorldSize(self):
        return self._WorldSize

    @WorldSize.setter
    def WorldSize(self, WSize):
        self._WorldSize = WSize

    @property
    def LocalRank(self):
        return self._LRank

    @LocalRank.setter
    def LocalRank(self, LRank):
        self._LRank = LRank

    @property
    def LocalSize(self):
        return self._LocalSize

    @LocalSize.setter
    def LocalSize(self, value):
        self._LocalSize = value

    @property
    def IPMaster(self):
        return self._IPMAddress

    @IPMaster.setter
    def IPMaster(self, value):
        self._IPMaster = value

    @property
    def Port(self):
        return self._Port

    @Port.setter
    def Port(self, value):
        self._Port = value

    @property
    def ParallelDirectory(self):
        return str(self._NFS)

    @ParallelDirectory.setter
    def ParallelDirectory(self):
        self._NFS = Path(value)

    @property
    def Parallel(self):
        return self._Parallel

    @Parallel.setter
    def Parallel(self, value):
        self._Parallel = value

    def __new__(cls):
        """! Overload __new__ method to intercept every creation of this
        object. When the object is already created we immediately return it.
        Thus we can call in any place in the code System().method and get the
        same view of our system without creating new objects.
        """
        if cls._instance is None:
            cls._instance = super(System, cls).__new__(cls)

        return cls._instance

    def __repr__(self):
        return (f'System(HostName={self.Name}, '
              f'ClusterName={self.ClusterName}, '
              f'Rank={self.Rank}, WorldSize={self.WorldSize}, '
              f'IPMaster={self.IPMaster} Port={self.Port}, '
              f'ParallelDirectory={self.ParallelDirectory}, '
              f'Parallel={self.Parallel}, '
              f'LocalRank={self.LocalRank}, LocalSize={self.LocalSize})')

def task(local_rank, rank, world_size):
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{local_rank}")
    print(f"I have device {device}")

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
    ddp_model = DDP(model, device_ids=[local_rank])
    
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
    


def main():
    mp.spawn(task,
        args=(world_size,),
        nprocs=world_size,
        join=True)


def setupDDPTorch():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = System().IPMaster
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(System().Port)
    dist.init_process_group( backend="nccl",
        world_size=System().WorldSize, rank=System().Rank)

    #We assume that the number of ranks per node is equal to the
    #numebr of GPUs in the system
    torch.cuda.set_device(System().LocalRank)


if __name__=="__main__":
    Sys = System()
    print(repr(Sys), flush=True)
    try:
        setupDDPTorch()
        task(Sys.LocalRank, Sys.Rank, Sys.WorldSize)
    finally:
        print("I am here")
        if dist.is_initialized():
            print("Cleaning up process group...")
            dist.destroy_process_group()
    sys.exit()


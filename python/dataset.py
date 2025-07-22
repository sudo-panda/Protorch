import torch
from torch_geometric.data import HeteroData, Dataset

class HecBenchDataset(Dataset):
    def __init__(self, file_list, device="cpu"):
        super(HecBenchDataset, self).__init__()
        self.device = device
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], "rb") as f:
            data: HeteroData = torch.load(f, weights_only=False)
            data.to(device=self.device)
        
        del data['module', 'symbol', 'value']
        del data['module']
        return data

class DevmapDataset(Dataset):
    def __init__(self, file_list, devmap_list, device="cpu"):
        super(DevmapDataset, self).__init__()
        self.device = device

        self.file_list = file_list
        self.devmap_list = devmap_list

        assert len(self.file_list) == len(self.devmap_list), f"File list and devmap list must have the same length, File List: {len(self.file_list)}, Devmap List: {len(self.devmap_list)}"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], "rb") as f:
            data: HeteroData = torch.load(f, weights_only=False)
            data.to(device=self.device)

        del data['module', 'symbol', 'value']
        del data['module']

        return data, int(self.devmap_list[idx] == "GPU")

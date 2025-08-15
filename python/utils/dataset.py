import torch
from torch_geometric.data import HeteroData, Dataset

class GraphDataset(Dataset):
    def __init__(self, file_list, device="cpu"):
        super(GraphDataset, self).__init__()
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

        data.file_path = self.file_list[idx]
        return data

class DevmapDataset(Dataset):
    def __init__(self, input_list, devmap_list, device="cpu"):
        super(DevmapDataset, self).__init__()
        self.device = device

        self.input_list = input_list
        self.devmap_list = devmap_list

        assert len(self.input_list) == len(self.devmap_list), f"File list and devmap list must have the same length, File List: {len(self.input_list)}, Devmap List: {len(self.devmap_list)}"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        with open(self.input_list[idx]["file_path"], "rb") as f:
            data: HeteroData = torch.load(f, weights_only=False)
            data.to(device=self.device)

        del data['module', 'symbol', 'value']
        del data['module']

        label = torch.Tensor([self.devmap_list[idx] == "GPU"]).to(device=self.device)

        for k, v in self.input_list[idx].items():
            data.__setattr__(k, v)
        return data, label

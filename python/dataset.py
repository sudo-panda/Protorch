import torch
from torch_geometric.data import HeteroData, Dataset

class FunctionGraphDataset(Dataset):
    def __init__(self, file_list, device="cpu"):
        super(FunctionGraphDataset, self).__init__()
        self.file_list = file_list
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], "rb") as f:
            data: HeteroData = torch.load(f, weights_only=False)
            data.to(device=self.device)
        
        del data['module', 'symbol', 'value']
        del data['module']
        return data





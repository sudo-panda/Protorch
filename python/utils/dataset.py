import ast
import torch
import torch.nn.functional as F
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
    def __init__(self, input_list, devmap_list, device="cpu", mean_std_dict=None):
        super(DevmapDataset, self).__init__()
        self.device = device

        self.input_list = input_list
        self.devmap_list = devmap_list

        if mean_std_dict is None:
            # Training
            stats = {}
            for item in self.input_list:
                for k, v in item.items():
                    if k == "file_path":
                        continue
                    
                    if k not in stats:
                        stats[k] = []
                    stats[k].append(v)

            mean_std_dict = {}
            for k, lst in stats.items():
                all_values = torch.tensor(lst, dtype=torch.float32)  # shape: [num_samples, feature_len]
                mean = all_values.mean(dim=0)
                std = all_values.std(dim=0)
                std = 1.0 if std == 0 else std
                mean_std_dict[k] = {"mean": mean, "std": std}

            self.mean_std_dict = mean_std_dict
        else: 
            # Validation / Test
            self.mean_std_dict = mean_std_dict


        assert len(self.input_list) == len(self.devmap_list), f"File list and devmap list must have the same length, File List: {len(self.input_list)}, Devmap List: {len(self.devmap_list)}"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):  # type: ignore
        with open(self.input_list[idx]["file_path"], "rb") as f:
            data: HeteroData = torch.load(f, weights_only=False)
            data.to(device=self.device)

        del data['module', 'symbol', 'value']
        del data['module']

        label = torch.Tensor([self.devmap_list[idx] == "GPU"]).to(device=self.device)

        for k, v in self.input_list[idx].items():
            if k != "file_path":
                mean = self.mean_std_dict[k]["mean"]
                std = self.mean_std_dict[k]["std"]
                v = (torch.tensor(v) - mean) / std
                v = v.to(self.device)
            data.__setattr__(k, v)

        return data, label


class VecParamsDataset(Dataset):
    def __init__(self, input_list, device="cpu"):
        super(VecParamsDataset, self).__init__()
        self.device = device

        self.input_list = input_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):  # type: ignore
        with open(self.input_list[idx]["file_path"], "rb") as f:
            data: HeteroData = torch.load(f, weights_only=False)
            data.to(device=self.device)

        del data['module', 'symbol', 'value']
        del data['module']
        del data['value', 'contains', 'value'] # FIXME: REMOVE

        data.file_path = self.input_list[idx]["file_path"]
        runtimes = torch.Tensor(list(ast.literal_eval(self.input_list[idx]["runtimes"]).values())).to(device=self.device, dtype=torch.float)
        log_labels = F.log_softmax(-5 * runtimes, dim=0)  # max(logits) = 0 internally
        labels = log_labels.exp()
        return data, labels
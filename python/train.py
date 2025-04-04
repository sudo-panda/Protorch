from dataset import FunctionGraphDataset
from pathlib import Path
import torch
from utils.common import get_adj_mat_from_edge_index
from GraMI.loss import GraMI_loss
from GraMI.model import GraMIModel

device = "cuda"

file_list = list(Path("/mnt/E/Workspaces/LLNL/HecBench/heterodatas/").glob("*.pt"))

dataset = FunctionGraphDataset(file_list, device=device)

model = GraMIModel(dataset[0][1], 16, 8)
model.to(device)
model.train()


optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def single_step(data):
    adj_mat = get_adj_mat_from_edge_index(data.x_dict, data.edge_index_dict)

    optimizer.zero_grad()

    X_hat, V, A, X_hat_prime, edge_logits, X_prime = model(data.x_dict, data.edge_index_dict, data["text"])

    assert torch.Tensor([X_hat[k].shape == X_hat_prime[k].shape for k in X_hat_prime.keys()]).all() == True
    assert torch.Tensor([data.x_dict[k].shape == X_prime[k].shape for k in X_prime.keys()]).all() == True
    assert torch.Tensor([adj_mat[k].shape == edge_logits[k].shape for k in data.edge_index_dict.keys()]).all() == True

    loss = GraMI_loss(data.x_dict, X_hat, adj_mat, V, A, edge_logits, X_hat_prime, X_prime)

    print(loss)
    loss.backward()
    optimizer.step()

index = 0
for file_name, data in dataset:
    index += 1
    if index < 151:
        continue
    print(file_name, "\n")
    single_step(data)

print(index)
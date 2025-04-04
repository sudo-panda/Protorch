from tqdm import tqdm
from dataset import FunctionGraphDataset

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common import get_adj_mat_from_edge_index
from GraMI.metrics import loss_fn, acc_fn
from GraMI.model import GraMIModel

from paths import GraMI_path, top_level_path

device = "cuda"
epochs =  1000
train_from_scratch = False


writer = SummaryWriter()

file_list = list((top_level_path / "HecBench" / "heterodatas").glob("*.pt"))

dataset = FunctionGraphDataset(file_list, device=device)

model = GraMIModel(dataset[0][1], 16, 8)

print(model)
if train_from_scratch and (GraMI_path / "latest.pt").exists():
    model.load_state_dict(torch.load(GraMI_path / "latest.pt"), strict=True)
model.to(device)
model.train()


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

def single_step(data):
    adj_mat = get_adj_mat_from_edge_index(data.x_dict, data.edge_index_dict)

    optimizer.zero_grad()

    X_hat, V, A, X_hat_prime, edge_logits, X_prime = model(data.x_dict, data.edge_index_dict, data["text"])

    assert torch.Tensor([X_hat[k].shape == X_hat_prime[k].shape for k in X_hat_prime.keys()]).all() == True
    assert torch.Tensor([data.x_dict[k].shape == X_prime[k].shape for k in X_prime.keys()]).all() == True
    assert torch.Tensor([adj_mat[k].shape == edge_logits[k].shape for k in data.edge_index_dict.keys()]).all() == True

    loss = loss_fn(data.x_dict, X_hat, adj_mat, V, A, edge_logits, X_hat_prime, X_prime)
    acc = acc_fn(data.x_dict, adj_mat, edge_logits, X_prime)

    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()

for i in range(epochs):
    print("Epoch:", i)
    index = 0
    tot_loss = 0
    tot_acc = 0

    for file_name, data in tqdm(dataset):
        loss, acc = single_step(data)
        tot_loss += loss
        tot_acc += acc
        index += 1

    torch.save(model.state_dict(), GraMI_path / "latest.pt")
    writer.add_scalar("Loss/train", tot_loss / index, i)
    writer.add_scalar("Acc/train", tot_acc / index, i)
    writer.flush()


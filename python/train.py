from tqdm import tqdm
from dataset import FunctionGraphDataset
from torch_geometric.loader import DataLoader

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common import get_adj_mat_from_edge_index, get_config
from GraMI.metrics import loss_fn, acc_fn
from GraMI.model import GraMIModel

from paths import GraMI_path, top_level_path

config = get_config()

device = config["device"]
epochs =  config["train"]["epochs"]
train_from_checkpoint = config["train"]["train_from_checkpoint"]
lr = config["train"]["learning_rate"]
decay = config["train"]["weight_decay"]
batch_size = config["train"]["batch_size"]


writer = SummaryWriter()

file_list = list((top_level_path / "HecBench" / "heterodatas").glob("*.pt"))[:10]

train_dataloader = DataLoader(FunctionGraphDataset(file_list, device=device), batch_size=2, shuffle=False)

data_sample = next(iter(train_dataloader))
model = GraMIModel(data_sample, 16, 8)

print(model)
if train_from_checkpoint and (GraMI_path / "latest.pt").exists():
    model.load_state_dict(torch.load(GraMI_path / "latest.pt"), strict=True)
model.to(device)
model.train()


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

def single_step(data, model):
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

    for batch in tqdm(train_dataloader):
        loss, acc = single_step(batch, model)
        tot_loss += loss
        tot_acc += acc
        index += 1

    torch.save(model.state_dict(), GraMI_path / "latest.pt")
    writer.add_scalar("Loss/train", tot_loss / index, i)
    writer.add_scalar("Acc/train", tot_acc / index, i)
    writer.flush()


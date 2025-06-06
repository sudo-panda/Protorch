from tqdm import tqdm
from dataset import FunctionGraphDataset
from torch_geometric.loader import DataLoader

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common import get_adj_mat_from_edge_index, device, epochs, get_log_dir_name, train_from_checkpoint, lr, decay, batch_size, parse_args
from GraMI.metrics import loss_fn, acc_fn
from GraMI.model import GraMIModel

from paths import GraMI_path, top_level_path

import sys

file_list = list((top_level_path / "HecBench" / "heterodatas").glob("*.pt"))[:10]
training_list = file_list[:int(len(file_list) * 0.6)]
validation_list = file_list[int(len(file_list) * 0.6):int(len(file_list) * 0.8)]
test_list = file_list[int(len(file_list) * 0.8):]
print(f"Training/Validation/Test split: {len(training_list)}/{len(validation_list)}/{len(test_list)}")

def single_step(data, model):
    adj_mat = get_adj_mat_from_edge_index(data.x_dict, data.edge_index_dict)

    X, X_hat, V, A, X_hat_prime, edge_logits, X_prime = model(data.x_dict, data.edge_index_dict, data["text"])

    assert torch.Tensor([X_hat[k].shape == X_hat_prime[k].shape for k in X_hat_prime.keys()]).all() == True
    assert torch.Tensor([X[k].shape == X_prime[k].shape for k in X_prime.keys()]).all() == True
    assert torch.Tensor([adj_mat[k].shape == edge_logits[k].shape for k in data.edge_index_dict.keys()]).all() == True

    loss = loss_fn(X, X_hat, adj_mat, V, A, edge_logits, X_hat_prime, X_prime)
    with torch.no_grad():
        acc = acc_fn(X, adj_mat, edge_logits, X_prime)

    return loss, acc

def main():
    train_dataloader = DataLoader(FunctionGraphDataset(training_list, device=device), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(FunctionGraphDataset(validation_list, device=device), batch_size=batch_size, shuffle=False)

    data_sample = next(iter(train_dataloader))
    
    args = parse_args()
    model = GraMIModel(data_sample, args)

    # print(model)
    if train_from_checkpoint and (GraMI_path / f"{args["model_name"]}.pt").exists():
        model.load_state_dict(torch.load(GraMI_path / f"{args["model_name"]}.pt"), strict=True)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    writer = SummaryWriter(log_dir=get_log_dir_name(args["model_name"]))

    for i in range(epochs):
        print("Epoch:", i)
        index_train = 0
        tot_train_loss = 0
        tot_train_acc = 0

        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            loss, acc = single_step(batch, model)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item() * batch.batch_size
            tot_train_acc += acc.item() * batch.batch_size
            index_train += batch.batch_size


        index_val = 0
        tot_val_loss = 0
        tot_val_acc = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                loss, acc = single_step(batch, model)
                tot_val_loss += loss.item() * batch.batch_size
                tot_val_acc += acc.item() * batch.batch_size
                index_val += batch.batch_size

        torch.save(model.state_dict(), GraMI_path / f"{args["model_name"]}.pt")
        writer.add_scalar("Loss/train", tot_train_loss / index_train, i)
        writer.add_scalar("Acc/train", tot_train_acc / index_train, i)
        writer.add_scalar("Loss/val", tot_val_loss / index_val, i)
        writer.add_scalar("Acc/val", tot_val_acc / index_val, i)
        writer.flush()
    
    writer.close()

if __name__ == "__main__":
    # python train.py --d_embed 10 --init_instr "64,32" --init_val "64,32" --init_num "" --init_typ "32" --init_attr "" --init_size "" --init_fdim 16 --enc_hgnn "16,16" --enc_mlp "64,16" --enc_fdim 8 --mname "latest" 
    main()
    print("Training complete.")

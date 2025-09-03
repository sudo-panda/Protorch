import torch
from torch_geometric.nn import global_add_pool


def pool_enc_outs(z_A, z_V, batch, batch_size, node_order):
    outs = [ z_A.sum(dim=1) ]
    for node_type in node_order:
        # FIXME: Can we make it faster?
        missing_batches = set(range(batch_size)) - set(batch[node_type].tolist())
        if missing_batches:
            for missing_batch in missing_batches:
                insert_idx = (batch[node_type] < missing_batch).sum().item()
                z_V[node_type] = torch.cat(
                    [
                        z_V[node_type][:insert_idx],
                        torch.zeros((1, z_V[node_type].size(1)), device=z_V[node_type].device),
                        z_V[node_type][insert_idx:]
                    ],
                    dim=0
                )
                batch[node_type] = torch.cat(
                    [
                        batch[node_type][:insert_idx],
                        torch.tensor([missing_batch], device=batch[node_type].device),
                        batch[node_type][insert_idx:]
                    ],
                    dim=0
                )
        
        outs.append(global_add_pool(z_V[node_type], batch[node_type]))
    return outs
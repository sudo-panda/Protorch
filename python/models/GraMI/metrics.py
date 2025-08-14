import torch.nn.functional as F
import torch
import numpy as np

def GraMI_loss(X, X_hat, adj_mat, V, A, edge_logits, 
               X_hat_prime, X_prime, variational, 
               lambdas=[2, 0.1, (1.0/15)], 
               betas=[1.0, 1.0]):
    loss_edge_mse = 0
    for k in adj_mat.keys():
        loss_edge_mse += F.binary_cross_entropy(edge_logits[k], adj_mat[k], reduction='mean')
    loss_edge_mse /= len(adj_mat.keys())

    loss_edge_kl = 0
    if variational:
        for k in V.keys():
            mean, log_var = V[k]
            loss_edge_kl += - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss_edge_kl /= len(V.keys())

    loss_edge = loss_edge_mse + betas[0] * loss_edge_kl
    # print(loss_edge, loss_edge_mse, 0.002 * loss_edge_kl)

    loss_attr_mse = 0
    for k in X_hat.keys():
        loss_attr_mse += F.mse_loss(X_hat_prime[k], X_hat[k], reduction='mean')
        # print(k, "\n", X_hat_prime[k], "\n", X_hat[k], "\n")

        if torch.isnan(X_hat_prime[k]).any() or torch.isinf(X_hat_prime[k]).any():
            print("Target contains NaN or Inf")

        if torch.isnan(X_hat[k]).any() or torch.isinf(X_hat[k]).any():
            print("Input contains NaN or Inf")
    loss_attr_mse /= len(X_hat.keys())

    loss_attr_kl = 0
    if variational:
        mean, log_var = A
        loss_attr_kl += - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    loss_attr = loss_attr_mse + betas[1] * loss_attr_kl
    # print(loss_attr, loss_attr_mse, 0.002 * loss_attr_kl)

    loss_rmse = 0
    for k in X.keys():
        loss_rmse += F.mse_loss(X_prime[k], X[k], reduction='mean')
    
        if torch.isnan(X_prime[k]).any() or torch.isinf(X_prime[k]).any():
            print("Target contains NaN or Inf")

    loss_rmse /= len(X.keys())
    loss_rmse = torch.sqrt(loss_rmse)
    
    # print(loss_rmse)
    loss = lambdas[0] * loss_edge + lambdas[1] * loss_attr + lambdas[2] * loss_rmse
    return loss

def edge_and_r2_acc(X, adj_mat, edge_logits, X_prime):
    """
    Compute:
      - edge_acc: average binary‐accuracy over all edge types
      - r2_attr:  average R² (coefficient of determination) over all feature types

    Args:
      X            (dict[str, Tensor]):   ground-truth features
      adj_mat      (dict[str, Tensor]):   ground-truth edge labels (0/1)
      edge_logits  (dict[str, Tensor]):   predicted logits for each edge type
      X_prime      (dict[str, Tensor]):   reconstructed features

    Returns:
      edge_acc     (Tensor): scalar in [0,1]
      r2_attr      (Tensor): scalar (can be negative if reconstruction is poor)
    """
    with torch.no_grad():
        # 1) Edge‐accuracy
        acc_values = []
        for k, labels in adj_mat.items():
            preds = (edge_logits[k] > 0.5).type(torch.float32)
            acc_k = (preds == labels).type(torch.float32).mean()
            acc_values.append(acc_k)
        if acc_values:
            edge_acc = torch.stack(acc_values).mean()
        else:
            print("Warning no edge_types in the adj_mat dictionary")
            edge_acc = torch.tensor(0.0)

        # 2) Attribute R²
        r2_vals = []
        for k, x in X.items():
            x_hat = X_prime[k]
            mse_k = F.mse_loss(x_hat, x, reduction='mean')
            var_k = x.var(unbiased=False)
            if var_k.item() < 1e-6:
                # skip near-constant features
                continue
            # add small fraction of var to avoid divide-by-zero
            r2_k = 1 - mse_k / (var_k + 1e-3 * var_k)
            r2_vals.append(r2_k)
        r2_attr = torch.stack(r2_vals).mean() if r2_vals else torch.tensor(0.)

        return np.array([edge_acc.item(), r2_attr.item()]) # send as numpy so that operators work downstream

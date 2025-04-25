import torch.nn.functional as F
import torch

def loss_fn(X, X_hat, adj_mat, V, A, edge_logits, X_hat_prime, X_prime, lambda0=0.5, lambda1=0.1, lambda2=(1.0/15)):
    loss_edge_mse = 0
    for k in adj_mat.keys():
        loss_edge_mse += F.binary_cross_entropy_with_logits(edge_logits[k], adj_mat[k], reduction='mean')
    loss_edge_mse /= len(adj_mat.keys())

    loss_edge_kl = 0
    for k in V.keys():
        mean, log_var = V[k]
        loss_edge_kl += - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss_edge_kl /= len(V.keys())

    loss_edge = loss_edge_mse + loss_edge_kl
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

    mean, log_var = A
    loss_attr_kl += - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    loss_attr = loss_attr_mse + loss_attr_kl
    # print(loss_attr, loss_attr_mse, 0.002 * loss_attr_kl)

    loss_rmse = 0
    for k in X.keys():
        loss_rmse += F.mse_loss(X_prime[k], X[k], reduction='mean')
    
        if torch.isnan(X_prime[k]).any() or torch.isinf(X_prime[k]).any():
            print("Target contains NaN or Inf")

    loss_rmse /= len(X.keys())
    loss_rmse = torch.sqrt(loss_rmse)
    
    # print(loss_rmse)
    loss = lambda0 * loss_edge + lambda1 * loss_attr + lambda2 * loss_rmse
    return loss

def acc_fn(X, adj_mat, edge_logits, X_prime):
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
            preds = (edge_logits[k] > 0.5).float()
            acc_k = (preds == labels.float()).float().mean()
            acc_values.append(acc_k)
        edge_acc = torch.stack(acc_values).mean() if acc_values else torch.tensor(0.0)
    
        # 2) Attribute R²
        r2_values = []
        for k, x in X.items():
            x_pred = X_prime[k]
            # MSE per element
            mse_k = F.mse_loss(x_pred, x, reduction='mean')
            # Variance of the reference
            var_k = x.var(unbiased=False)
            # R², with eps to avoid div-by-zero
            r2_k = 1 - mse_k / (var_k + 1e-8)
            r2_values.append(r2_k)
        r2_attr = torch.stack(r2_values).mean() if r2_values else torch.tensor(0.0)
    
        return (edge_acc + r2_attr) / 2
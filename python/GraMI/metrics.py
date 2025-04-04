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
    with torch.no_grad():
        acc_edge_mse = 1
        for k in adj_mat.keys():
            acc_edge_mse += torch.exp(-F.binary_cross_entropy_with_logits(edge_logits[k], adj_mat[k], reduction='sum'))
        acc_edge_mse /= len(adj_mat.keys())
        
        acc_rmse = 0
        for k in X.keys():
            acc_rmse += torch.exp(-F.mse_loss(X_prime[k], X[k], reduction='sum'))
        acc_rmse /= len(X.keys())
        
        acc = (acc_edge_mse + acc_rmse) / 2
        return acc
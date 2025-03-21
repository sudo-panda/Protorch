import torch.nn.functional as F
import torch

def GraMI_loss(X, X_hat, adj_mat, V, A, edge_logits, X_hat_prime, X_prime, lambda1=1, lambda2=1):
    loss_edge = 0
    for k in adj_mat.keys():
        loss_edge += F.binary_cross_entropy_with_logits(edge_logits[k], adj_mat[k], reduction='mean')
    loss_edge /= len(adj_mat.keys())

    print(loss_edge)
    for k in V.keys():
        mean, log_var = V[k]
        loss_edge += - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    loss_attr = 0
    for k in X_hat.keys():
        loss_attr += F.mse_loss(X_hat_prime[k], X_hat[k], reduction='mean')

        if torch.isnan(X_hat_prime[k]).any() or torch.isinf(X_hat_prime[k]).any():
            print("Target contains NaN or Inf")

        if torch.isnan(X_hat[k]).any() or torch.isinf(X_hat[k]).any():
            print("Input contains NaN or Inf")
    loss_attr /= len(X_hat.keys())

    mean, log_var = A
    loss_attr += - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    print(loss_attr)
    loss_rmse = 0
    for k in X.keys():
        loss_rmse += F.mse_loss(X_prime[k], X[k], reduction='sum')
    
        if torch.isnan(X_prime[k]).any() or torch.isinf(X_prime[k]).any():
            print("Target contains NaN or Inf")

    loss_rmse /= len(X.keys())
    loss_rmse = torch.sqrt(loss_rmse)
    
    print(loss_rmse)
    loss = loss_edge + lambda1 * loss_attr + lambda2 * loss_rmse
    return loss
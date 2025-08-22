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

def edge_and_r2_acc(X, adj_mat, edge_logits, X_prime) -> np.ndarray:
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
            mse_k_list = []
            for x_hat_ele in x_hat:
                mse_k_list.append(F.mse_loss(x_hat_ele, x, reduction='mean'))
            mse_k = torch.stack(mse_k_list).mean()
            var_k = x.var(unbiased=False)
            if var_k.item() < 1e-6:
                # skip near-constant features
                continue
            # add small fraction of var to avoid divide-by-zero
            r2_k = 1 - mse_k / (var_k + 1e-3 * var_k)
            r2_vals.append(r2_k)
        r2_attr = torch.stack(r2_vals).mean() if r2_vals else torch.tensor(0.)

        return np.array([edge_acc.item(), r2_attr.item()]) # send as numpy so that operators work downstream

def loss_function_a_mse(n_A_mu, n_A_logvar, z_A, eps_A):
    SMALL = 1e-6

    n_A_std = torch.exp(0.5 * n_A_logvar)
    B, J, N, zdim = z_A.shape
    K = n_A_mu.shape[1] - J

    mu_mix, mu_emb = n_A_mu[:, :K, :], n_A_mu[:, K:, :]
    std_mix, std_emb = n_A_std[:, :K, :], n_A_std[:, K:, :]

    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * z_A.pow(2), dim=[1, 2]).mean()

    # compute log_posterior
    # Z.shape = [B, J, 1, N, zdim]
    Z = z_A.view(B, J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [B, 1, K, N, zdim]
    mu_mix = mu_mix.view(B, 1, K, N, zdim)
    std_mix = std_mix.view(B, 1, K, N, zdim)

    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [B, J, K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2, -1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2, -1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [B, J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps_A.pow(2), dim=[-2, -1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J = log_post_ker_J.view(B, -1, 1)

    # assert False, f"{log_post_ker_J.shape}  {log_post_ker_JK.shape}"

    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()

    return log_prior_ker, log_posterior_ker

def loss_distribution(node_mu, node_logvar, z_node, eps_node): # For each node type
    SMALL = 1e-6
    std = torch.exp(0.5 * node_logvar)
    J, N, zdim = z_node.shape
    K = node_mu.shape[0] - J

    mu_mix, mu_emb = node_mu[:K, :], node_mu[K:, :]
    std_mix, std_emb = std[:K, :], std[K:, :]

    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * z_node.pow(2), dim=[-2, -1]).mean()
    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z = z_node.view(J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix = mu_mix.view(1, K, N, zdim)
    std_mix = std_mix.view(1, K, N, zdim)

    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J, K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2, -1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2, -1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps_node.pow(2), dim=[-2, -1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J = log_post_ker_J.view(-1, 1)

    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()
    return log_prior_ker, log_posterior_ker


def get_mseloss_score(x, x_rec):
    def get_rec(pred, orig):
        loss_ac = F.mse_loss(pred, orig, reduction="sum")
        loss_ac = torch.sqrt(loss_ac)
        return loss_ac

    total_cost = []
    for node_type in x.keys():
        x_node = x[node_type]
        x_rec_node = x_rec[node_type]
        rec_costs = torch.stack(
            [get_rec(pred, x_node) for pred in torch.unbind(x_rec_node, dim=0)]
        )
        rec_cost = rec_costs.mean()
        total_cost.append(rec_cost)

    total_cost = torch.stack(total_cost)

    return total_cost.mean()

def get_pos_norm(adj):
    assert adj.ndim == 2
    SMALL = 1e-8
    adj = adj.detach().clone()
    adj_pos_ele = adj.sum()
    total_ele = adj.shape[0] * adj.shape[1]
    pos_weight = (total_ele - adj_pos_ele) / (adj_pos_ele + SMALL)
    norm = (total_ele / float((total_ele - adj_pos_ele) * 2))
    return pos_weight, norm

def loss_get_rec(edge_logits, labels, norm, pos_weight):
    def get_rec(pred):
        log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))
        rec = -log_lik.mean()
        return rec

    SMALL = 1e-6
    edge_logits = torch.clamp(edge_logits, min=SMALL, max=1 - SMALL)

    # compute rec_cost
    rec_costs = torch.stack(
        [get_rec(pred) for pred in torch.unbind(edge_logits, dim=0)],
        dim=0)
    # average over J * N * N items
    rec_cost = rec_costs.mean()
    return rec_cost

def SeeR_GraMI_loss(x, x_tile, adj_mat, n_V, n_A, z_A, z_V, eps_A, eps_V, edge_logits, x_tile_rec, x_rec,
            epoch, lambdas=[1, 0.1, 0.1]):
    loss_rec_all, loss_prior_all, loss_post_all = [], [], []

    for edge_type in adj_mat.keys():
        recovered = edge_logits[edge_type]
        adj_label = adj_mat[edge_type]
        pos_weight, norm = get_pos_norm(adj_label)
        loss_rec_all.append(loss_get_rec(recovered, adj_label, norm, pos_weight))

    n_nodes = 0
    for node_type in n_V.keys():
        node_mu, node_logvar, z_node, eps_node = \
            n_V[node_type][0], n_V[node_type][1], z_V[node_type], eps_V[node_type]
        #   n_V.mu             n_V.logvar         z_V             eps_V
        n_nodes += node_mu.shape[1]

        loss_prior, loss_post = loss_distribution(node_mu, node_logvar, z_node, eps_node)
        loss_prior_all.append(loss_prior)
        loss_post_all.append(loss_post)

    loss_prior_f, loss_post_f = loss_function_a_mse(
        n_A[0], # mu
        n_A[1], # logvar
        z_A=z_A,
        eps_A=eps_A
    )
    loss_post_all.append(loss_post_f)
    loss_prior_all.append(loss_prior_f)

    loss_rec_f = get_mseloss_score(x_tile, x_tile_rec)

    # loss_recover = get_mseloss_score(mask=mask_fea, fea_orig=fea_train, fea_pred=x_rec)
    loss_recover = get_mseloss_score(x, x_rec)

    loss_rec = sum(loss_rec_all)
    loss_post = sum(loss_post_all)
    loss_prior = sum(loss_prior_all)

    WU = np.min([epoch / 300., 1.])
    reg = (loss_post - loss_prior) / (n_nodes ** 2)
    # reg = (loss_post - loss_prior) * WU / (n_nodes ** 2)
    loss_train = loss_rec * lambdas[0] + WU * reg + lambdas[1] * loss_rec_f + lambdas[2] * loss_recover

    return loss_train

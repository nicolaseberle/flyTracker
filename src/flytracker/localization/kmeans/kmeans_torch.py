import torch
from torch.nn.functional import normalize


def kmeans_torch(X, init, tol=1e-4, device="cuda"):
    def step(X, mu):
        # E step
        dist_matrix = torch.cdist(X, mu)
        labels = torch.argmin(dist_matrix, axis=1)

        # M step
        M.zero_()  # resetting
        M[labels, torch.arange(n_samples)] = 1.0  #  updating in place
        mu = torch.matmul(normalize(M, p=1, dim=1), X)
        return mu, labels

    n_samples, n_clusters = X.shape[0], init.shape[0]
    M = torch.zeros((n_clusters, n_samples)).to(device, non_blocking=True)

    new_centers, old_centers = step(X, init)[0], init
    while torch.linalg.norm(new_centers - old_centers) > tol:
        new_centers, old_centers = step(X, new_centers)[0], new_centers
    return new_centers


import torch
from torch.nn.functional import normalize
from functools import partial


def kmeans_torch(tol=1e-4, device="cuda"):
    def kmeans(X, init):
        n_samples, n_clusters = X.shape[0], init.shape[0]
        M = torch.zeros((n_clusters, n_samples), device=device)
        update = partial(EM_step, X, M)

        new_centers, old_centers = update(init)[0], init
        while torch.linalg.norm(new_centers - old_centers) > tol:
            (new_centers, labels), old_centers = update(new_centers), new_centers
        return new_centers, labels

    def EM_step(X, M, mu):
        # E step
        dist_matrix = torch.cdist(X, mu)
        labels = torch.argmin(dist_matrix, axis=1)

        # M step
        M.zero_()  # resetting
        M[labels, torch.arange(M.shape[-1])] = 1.0  #  updating in place
        mu = torch.matmul(normalize(M, p=1, dim=1), X)
        return mu, labels

    return kmeans

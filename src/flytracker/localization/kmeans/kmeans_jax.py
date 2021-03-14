import jax
from jax import jit, lax, numpy as jnp
from functools import partial


@jit
def cdist(x, y):
    return jnp.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


@jit
def kmeans_jax(X, init):
    def cond_fun(carry):
        new_centers, old_centers = carry
        return jnp.linalg.norm(new_centers - old_centers) > 1e-4

    def kmeans_step(X, mu):
        # E step
        dist_matrix = cdist(X, mu)
        labels = jnp.argmin(dist_matrix, axis=1)

        # Mstep
        M = jnp.zeros((n_clusters, n_samples))
        M = jax.ops.index_update(M, (labels, jnp.arange(n_samples)), 1.0)
        new_mu = (M / jnp.sum(M, axis=1, keepdims=True)) @ X
        return new_mu, labels

    def body_fun(carry):
        new, _ = carry
        return step(new)[0], new

    n_clusters = init.shape[0]
    n_samples = X.shape[0]
    step = partial(kmeans_step, X)

    init_carry = (step(init)[0], init)
    mu, _ = lax.while_loop(cond_fun, body_fun, init_carry)
    _, labels = step(mu)
    return mu, labels
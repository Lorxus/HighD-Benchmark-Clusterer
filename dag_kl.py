import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import ops
from jax.scipy import special
jax.default_device(jax.devices('cpu')[0])

def get_cov_test(n: int, alpha: float) -> jax.Array:
    ones_mat = jnp.eye(n+1)
    ones_mat = ones_mat.at[:, 0].set(-1)
    ones_mat = ones_mat.at[0, 0].set(1)

    alpha_mat = jnp.eye(n+1) * alpha
    alpha_mat = alpha_mat.at[0, 0].set(1)

    ones_inv = jnp.linalg.inv(ones_mat)

    cov = (ones_inv @ alpha_mat @ alpha_mat @ ones_inv.T)
    return cov


def get_factored_s(s: jax.Array, parent_list: list[list[int]]) -> jax.Array:
    assert len(parent_list) == len(s)
    parent_list = [jnp.array(p, dtype=jnp.int32) for p in parent_list]

    cov = jnp.linalg.inv(s)
    coeffs = [jnp.linalg.solve(cov[parents, :][:, parents], cov[i, parents]) for i, parents in enumerate(parent_list)]
    schurs = jnp.array([cov[i, i] - coeffs[i] @ cov[parents, i] for i, parents in enumerate(parent_list)])
    L = jnp.eye(len(s))
    for i, parents in enumerate(parent_list):
        L = L.at[i, parents].set(-coeffs[i])
    L = jnp.diag(1.0/jnp.sqrt(schurs)) @ L

    s_dag = L.T @ L
    return s_dag


def dag_kl(s: jax.Array, parent_list: list[list[int]]) -> float:
    s_dag = get_factored_s(s, parent_list)
    return kl_divergence(s, s_dag)


def kl_divergence(precision_p, precision_q):
    """
    Calculate the KL divergence between two normal distributions represented as precision matrices.

    Args:
        p (jax.numpy.ndarray): Precision matrix of the first distribution.
        q (jax.numpy.ndarray): Precision matrix of the second distribution.

    Returns:
        float: KL divergence between the two distributions.
    """
    precision_q = jnp.array(precision_q, dtype=jnp.float64)
    precision_p = jnp.array(precision_p, dtype=jnp.float64)
    # Get the dimensionality of the distributions
    d = precision_p.shape[0]

    # Calculate the covariance matrix from the precision matrix
    cov_p = jnp.linalg.inv(precision_p)

    # Calculate the trace term
    trace_term = jnp.sum(precision_q * cov_p)

    # Calculate the log determinant term
    log_det_term = jnp.linalg.slogdet(precision_p)[1] - jnp.linalg.slogdet(precision_q)[1]

    # Calculate the KL divergence
    kl_div = 0.5 * (trace_term + log_det_term - d)

    return kl_div/jnp.log(2.0)


if __name__ == '__main__':
    cov = get_cov_test(5, 0.5)
    s_full = jnp.linalg.inv(cov)
    s_x = jnp.linalg.inv(cov[1:, 1:])

    s = s_full

    print("Should be 0:", kl_divergence(s, s))
    print("Should be positive:", kl_divergence(s, s + jnp.eye(6)))

    print("Should be zero:", dag_kl(s, [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]))
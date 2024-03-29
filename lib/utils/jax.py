import jax.numpy as jnp
import jax.random as jr
from jax import tree_map, vmap


### functions ###
def expsum(a, axis=0):
    """
    Numerically stabilized
    """
    a_max = a.max(axis=axis, keepdims=True)
    return jnp.exp(a_max.sum(axis=axis)) * jnp.exp((a - a_max).sum(axis=axis))


def safe_log(x, eps=1e-10):
    return jnp.log(x + eps)  # avoid divergence at 0


def safe_sqrt(x, eps=1e-10):
    return jnp.sqrt(x + eps)  # avoid gradient divergence at 0


def softplus(x):
    return jnp.log(1 + jnp.exp(-jnp.abs(x))) + jnp.maximum(
        x, 0
    )  # numerically stabilized


def softplus_inv(x):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    return jnp.log(1 - jnp.exp(-jnp.abs(x))) + jnp.maximum(
        x, 0
    )  # numerically stabilized


def rectquad(x):
    """
    Rectified quadratic
    """
    return 0.5 * (x + jnp.sqrt(x**2 + 4))


def rectquad_inv(x):
    """
    Rectified quadratic inverse
    """
    return x - 1 / x


def constrain_diagonal(K, lower_lim=1e-6):
    """
    Enforce matrix K has diagonal elements with lower limit
    """
    K_diag = jnp.diag(jnp.diag(K))
    K = jnp.where(jnp.any(jnp.diag(K) < 0), jnp.where(K_diag < 0, lower_lim, K_diag), K)
    return K


### Monte Carlo ###
def mc_sample(sample_shape, key, approx_points):
    z = jr.normal(key, shape=(*sample_shape, approx_points))
    w = jnp.ones((approx_points,)) / approx_points
    return z, w


def percentiles_from_samples(
    samples, percentiles=[0.05, 0.5, 0.95], smooth_length=1, conv_mode="same"
):
    """
    Compute quantile intervals from samples, samples has shape (sample_dim, event_dims..., T).

    :param torch.tensor samples: input samples of shape (MC, num_points, dims)
    :param list percentiles: list of percentile values to look at
    :param int smooth_length: time steps over which to smooth with uniform block
    :returns: list of tensors representing percentile boundaries
    :rtype: list
    """
    num_samples, T = samples.shape[0], samples.shape[-1]

    samples = jnp.sort(samples, axis=0)
    percentile_samples = [
        samples[int(num_samples * percentile)] for percentile in percentiles
    ]

    if smooth_length > 1:  # smooth the samples
        window = 1.0 / smooth_length * jnp.ones(smooth_length)
        percentile_samples = [
            vmap(jnp.convolve, (1, None, None), 1)(percentile_sample, window, conv_mode)
            for percentile_sample in percentile_samples
        ]

    return percentile_samples

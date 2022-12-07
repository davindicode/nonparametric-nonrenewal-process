import jax.numpy as jnp
from jax import random, tree_map, vmap


### pytree ###
def copy_pytree(_pytree):
    """
    None values are not regarded as leaves, so are skipped
    """
    return tree_map(lambda x: jnp.array(x), _pytree)


### functions ###
def expsum(a, axis=0):
    """
    Numerically stabilized
    """
    a_max = a.max(axis=axis, keepdims=True)
    return jnp.exp(a_max.sum(axis=axis)) * jnp.exp((a - a_max).sum(axis=axis))


def softplus_list(x_):
    """
    Softplus positivity mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = [jnp.log(1 + jnp.exp(-jnp.abs(x_[0]))) + jnp.maximum(x_[0], 0)]
    for i in range(1, len(x_)):
        if x_[i] is not []:
            y_ = y_ + [jnp.log(1 + jnp.exp(-jnp.abs(x_[i]))) + jnp.maximum(x_[i], 0)]
    return y_


def softplus_inv_list(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = x_
    for i in range(len(x_)):
        if x_[i] is not []:
            y_[i] = jnp.log(1 - jnp.exp(-jnp.abs(x_[i]))) + jnp.maximum(x_[i], 0)
    return y_


def softplus(x):
    return jnp.log(1 + jnp.exp(-jnp.abs(x))) + jnp.maximum(
        x, 0
    )  # numerically stabilized


def sigmoid(x):
    return jnp.exp(x) / (jnp.exp(x) + 1.0)


def softplus_inv(x):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    if x is None:
        return x
    else:
        return jnp.log(1 - jnp.exp(-jnp.abs(x))) + jnp.maximum(
            x, 0
        )  # numerically stabilized


### Monte Carlo ###
def sample_gaussian_noise(key, mean, Lcov):
    gaussian_sample = mean + Lcov @ random.normal(key, shape=mean.shape)
    return gaussian_sample


def mc_sample(dim, key, approx_points):
    z = random.normal(key, shape=(dim, approx_points))
    w = np.ones((approx_points,)) / approx_points
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

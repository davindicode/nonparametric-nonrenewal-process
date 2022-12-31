import math
from functools import partial


import jax.numpy as np

from jax.scipy.special import erf, gammaln

from ..utils.jax import expsum, mc_sample, sigmoid, softplus, softplus_inv
from ..utils.linalg import gauss_hermite, get_blocks, inv

from .factorized import Gaussian, ZeroInflatedPoisson, NegativeBinomial, ConwayMaxwellPoisson

_log_twopi = math.log(2 * math.pi)



class HeteroscedasticGaussian(Gaussian):
    """
    Heteroscedastic Gaussian likelihood
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """

    def __init__(self, out_dims, link="softplus", array_type=jnp.float32):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__(out_dims, 2 * out_dims)
        if link == "exp":
            self.link_fn = lambda x: np.exp(x)
            self.dlink_fn = lambda x: np.exp(x)
        elif link == "softplus":
            self.link_fn = lambda x: softplus(x)
            self.dlink_fn = lambda x: sigmoid(x)
        else:
            raise NotImplementedError("link function not implemented")

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp):
        """
        Evaluate the log-likelihood
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[0], np.maximum(self.link_fn(f[1]) ** 2, 1e-8)
        ll = -0.5 * ((_log_twopi + np.log(var)) + (y - mu) ** 2 / var)
        return ll
    
    
    
    

class HeteroscedasticZeroInflatedPoisson(ZeroInflatedPoisson):
    """
    Heteroscedastic ZIP
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        array_type=jnp.float32, 
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, None, tensor_type, strict_likelihood)
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = torch.sigmoid

    def constrain(self):
        return
    


class HeteroscedasticNegativeBinomial(NegativeBinomial):
    """
    Heteroscedastic NB
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        array_type=jnp.float32, 
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, None, tensor_type, strict_likelihood)
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = torch.nn.functional.softplus

    def constrain(self):
        return


class HeteroscedasticConwayMaxwellPoisson(ConwayMaxwellPoisson):
    """
    Heteroscedastic CMP
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        array_type=jnp.float32, 
        J=100,
        strict_likelihood=True,
    ):
        super().__init__(
            tbin, neurons, inv_link, None, tensor_type, J, strict_likelihood
        )
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = lambda x: x

    def constrain(self):
        return

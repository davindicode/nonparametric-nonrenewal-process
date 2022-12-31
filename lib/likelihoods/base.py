from typing import Any, Callable

import jax.numpy as jnp

from ..base import module




### classes ###
class Likelihood(module):
    
    f_dims: int
    out_dims: int
    link_type: str
    link_fn: Union[Callable, None]
        
    def __init__(self, obs_dims, f_dims, link_type, array_type):
        """
        all hyperparameters stored are pre-transformation
        :param hyp: (hyper)parameters of the likelihood model
        """
        super().__init__(array_type)
        self.f_dims = f_dims
        self.out_dims = out_dims
        # num_f_per_out = self.f_dims // self.out_dims
        
        self.link_type = link_type
        if link_type == "log":
            self.inverse_link = lambda x: jnp.exp(x)
            #self.dlink_fn = lambda mu: np.exp(mu)
        elif link_type == "logit":
            self.inverse_link = lambda x: softplus(x)
            #self.dlink_fn = lambda mu: sigmoid(mu)
        elif link_type == "rectified":
            self.inverse_link = lambda x: jnp.maximum(x, 0.)
        elif link_type == "none":
            self.inverse_link = None
        else:
            raise NotImplementedError("link function not implemented")




class FactorizedLikelihood(Likelihood):
    """
    The likelihood model class, p(yₙ|fₙ) factorized across time points
    fₙ can be a vector (multiple parameters per observation)

    variational_expectation() computes all E_q(f) related quantities and its derivatives for NGD
    We allow multiple f (vector fₙ) to correspond to a single yₙ as in heteroscedastic likelihoods

    The default functions here use cubature/MC approximation methods, exact integration is specific
    to certain likelihood classes.
    """
    
    def __init__(self, obs_dims, f_dims, link_type, array_type):
        super().__init__(obs_dims, f_dims, link_type, array_type)
#     def grads_log_likelihood_n(self, f_mean, df_points, y, lik_params, derivatives):
#         """
#         Factorization over data points n, vmap over out_dims
#         vmap over approx_points

#         :param np.array f_mean: (scalar) or (cubature,) for cubature_dim > 1
#         :param np.array df_points: (approx_points,) or (cubature, approx_points) for cubature_dim > 1

#         :return:
#             expected log likelihood ll (approx_points,)
#             dll_dm (approx_points,) or (cubature, approx_points)
#             d2ll_dm2 (approx_points,) or (cubature, cubature, approx_points)
#         """
#         f = (
#             df_points + f_mean
#         )  # (approx_points,) or (cubature, approx_points) for cubature_dim > 1

#         if derivatives:

#             def grad_func(f):
#                 ll, dll_dm = value_and_grad(self.log_likelihood_n, argnums=0)(
#                     f, y, lik_params
#                 )
#                 return dll_dm, (ll, dll_dm)

#             def temp_func(f):
#                 # dll_dm, (ll,) = grad_func(f)
#                 d2ll_dm2, aux = jacrev(grad_func, argnums=0, has_aux=True)(f)
#                 ll, dll_dm = aux
#                 return ll, dll_dm, d2ll_dm2

#             ll, dll_dm, d2ll_dm2 = vmap(temp_func, in_axes=0, out_axes=(0, 0, 0))(f)

#         else:
#             ll = vmap(self.log_likelihood_n, (0, None, None))(f, y, lik_params)
#             dll_dm, d2ll_dm2 = None, None

#         return ll, dll_dm, d2ll_dm2

    def log_likelihood(self, f, y, lik_params):
        raise NotImplementedError(
            "direct evaluation of this log-likelihood is not implemented"
        )

    def variational_expectation(
        self, prng_state, jitter, y, f_mean, f_cov,
    ):
        """
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ and its derivatives
        The log marginal likelihood is log E[p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        onlt the block-diagonal part of f_std matters
        :param np.array f_mean: mean of q(f) of shape (f_dims,)
        :param np.array f_cov: covariance of q(f) of shape (f_dims, f_dims)
        :return:
            log likelihood: expected log likelihood
            dlambda_1: gradient of E_q(f)[ log p(y|f) ] w.r.t. mean natural parameter
            dlambda_2: gradient of E_q(f)[ log p(y|f) ] w.r.t. covariance natural parameter
        """
        cubature_dim = self.num_f_per_out  # use smaller subgrid for cubature
        f, w = self.approx_int_func(cubature_dim, prng_state)

        ### compute transformed f locations ###
        # turn f_cov into lower triangular block diagonal matrix f_
        if cubature_dim == 1:
            f = np.tile(
                f, (self.out_dims, 1)
            )  # copy over out_dims, (out_dims, cubature_dim)
            f_var = np.diag(f_cov)
            f_std = np.sqrt(f_var)
            f_mean = f_mean[:, None]  # (f_dims, 1)
            df_points = f_std[:, None] * f  # (out_dims, approx_points)

        else:  # block-diagonal form
            f = np.tile(
                f[None, ...], (self.out_dims, 1, 1)
            )  # copy subgrid (out_dims, cubature_dim, approx_points)
            f_cov = get_blocks(np.diag(np.diag(f_cov)), self.out_dims, cubature_dim)
            # chol_f_cov = np.sqrt(np.maximum(f_cov, 1e-12)) # diagonal, more stable
            chol_f_cov = cholesky(
                f_cov + jitter * np.eye(cubature_dim)[None, ...]
            )  # (out_dims, cubature_dim, cubature_dim)

            f_mean = f_mean.reshape(self.out_dims, cubature_dim, 1)
            df_points = chol_f_cov @ f  # (out_dims, cubature_dim, approx_points)

        ### derivatives ###
        in_shape = tree_map(lambda x: 0, lik_params)
        if derivatives:
            ll, dll_dm, d2ll_dm2 = vmap(
                self.grads_log_likelihood_n,
                in_axes=(0, 0, 0, in_shape, None),
                out_axes=(0, 0, 0),
            )(
                f_mean, df_points, y, lik_params, True
            )  # vmap over out_dims

            if mask is not None:  # apply mask
                dll_dm = np.where(
                    mask[:, None], 0.0, dll_dm
                )  # (out_dims, approx_points)
                d2ll_dm2 = np.where(
                    mask[:, None], 0.0, d2ll_dm2
                )  # (out_dims, approx_points)

            dEll_dm = (w[None, :] * dll_dm).sum(1)
            d2Ell_dm2 = (w[None, :] * d2ll_dm2).sum(1)

            if cubature_dim == 1:  # only need diagonal f_cov
                dEll_dV = 0.5 * d2Ell_dm2
                dlambda_1 = (dEll_dm - 2 * (dEll_dV * f_mean[:, 0]))[
                    :, None
                ]  # (f_dims, 1)
                dlambda_2 = np.diag(dEll_dV)  # (f_dims, f_dims)

            else:
                dEll_dV = 0.5 * d2Ell_dm2[..., 0]
                dlambda_1 = dEll_dm[:, None] - 2 * (dEll_dV @ f_mean).reshape(
                    -1, 1
                )  # (f_dims, 1)
                dlambda_2 = dEll_dV  # (f_dims, f_dims)

        else:  # only compute log likelihood
            ll, dll_dm, d2ll_dm2 = vmap(
                self.grads_log_likelihood_n,
                in_axes=(0, 0, 0, in_shape, None),
                out_axes=(0, 0, 0),
            )(
                f_mean, df_points, y, lik_params, False
            )  # vmap over n
            dlambda_1, dlambda_2 = None, None

        ### expected log likelihood ###
        # f_mean and f_cov are from P_smoother
        if mask is not None:  # apply mask
            ll = np.where(mask[:, None], 0.0, ll)  # (out_dims, approx_points)
        weighted_log_lik = w * ll.sum(0)  # (approx_pts,)
        E_log_lik = weighted_log_lik.sum()  # E_q(f)[log p(y|f)]

        return E_log_lik, dlambda_1, dlambda_2

        
        
class CountLikelihood(FactorizedLikelihood):
    """
    For handling count data
    """
    tbin: float
    
    def __init__(self, out_dims, f_dims, tbin, link_type, array_type):
        super().__init__(out_dims, f_dims, link_type, array_type)
        self.tbin = tbin

        
        
        
class RenewalLikelihood(Likelihood):
    """
    Renewal model base class
    """
    dt: float

    def __init__(
        self, out_dims, f_dims, dt, link_fn, array_type, 
    ):
        super().__init__(obs_dims, f_dims, link_type, array_type)
        self.dt = dt

    def log_likelihood(self, spiketimes, pre_rates, covariates, neuron, num_ISIs):
        """
        Ignore the end points of the spike train
        
        :param jnp.ndarray pre_rates: pre-link rates (mc, out_dims, ts)
        :param List spiketimes: list of spike time indices arrays per neuron
        :param jnp.ndarray covariates: covariates time series (mc, out_dims, ts, in_dims)
        """
        mc, ts = covariates.shape[0], covariates.shape[2]
        
        # map posterior samples
        rates = self.link_fn(pre_rates)
        taus = self.dt * jnp.cumsum(rates, axis=2)
        
        # rate rescaling
        rISI = jnp.empty((mc, self.out_dims, num_ISIs))
        
        for en, spkinds in enumerate(spiketimes):
            isi_count = jnp.maximum(spkinds.shape[0] - 1, 0)
            
            def body(i, val):
                val[:, en, i] = taus[:, i]
                return val
            
            rISI[:, en, :] = lax.fori_loop(0, isi_count, body, rISI[:, en, :])
            
        # NLL
        log_renewals = jnp.nansum(self.log_renewal_density(rISI), axis=2)  # (mc, out_dims)
        
        ll = log_rates + log_renewals
        return ll
    

    def sample_helper(self, h, b, neuron, scale, samples):
        """
        MC estimator for NLL function.

        :param torch.Tensor scale: additional scaling of the rate rescaling to preserve the ISI mean

        :returns: tuple of rates, spikes*log(rates*scale), rescaled ISIs
        :rtype: tuple
        """
        batch_edge, _, _ = self.batch_info
        scale = scale.expand(1, self.F_dims)[
            :, neuron, None
        ]  # rescale to get mean 1 in renewal distribution
        rates = self.f(h) * scale
        spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
            self.dt.device
        )
        # self.spikes[b][:, neuron, self.filter_len-1:]
        if (
            self.trials != 1 and samples > 1 and self.trials < h.shape[0]
        ):  # cannot rely on broadcasting
            spikes = spikes.repeat(
                samples, 1, 1
            )  # trial blocks are preserved, concatenated in first dim

        if (
            self.inv_link == "exp"
        ):  # bit masking seems faster than integer indexing using spiketimes
            n_l_rates = (spikes * (h + torch.log(scale))).sum(-1)
        else:
            n_l_rates = (spikes * torch.log(rates + 1e-12)).sum(
                -1
            )  # rates include scaling

        spiketimes = [[s.to(self.dt.device) for s in ss] for ss in self.spiketimes[b]]
        rISI = self.rate_rescale(neuron, spiketimes, rates, self.duplicate[b])
        return rates, n_l_rates, rISI

    def objective(self, F_mu, F_var, XZ, b, neuron, scale, samples=10, mode="MC"):
        """
        :param torch.Tensor F_mu: model output F mean values of shape (samplesxtrials, neurons, time)

        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if mode == "MC":
            h = self.mc_gen(F_mu, F_var, samples, neuron)
            rates, n_l_rates, rISI = self.sample_helper(h, b, neuron, scale, samples)
            ws = torch.tensor(1.0 / rates.shape[0])
        elif mode == "direct":
            rates, n_l_rates, spikes = self.sample_helper(
                F_mu[:, neuron, :], b, neuron, samples
            )
            ws = torch.tensor(1.0 / rates.shape[0])
        else:
            raise NotImplementedError

        return self.nll(n_l_rates, rISI, neuron), ws

    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample spike trains from the modulated renewal process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        spiketimes = gen_IRP(
            self.ISI_dist(neuron), rate[:, neuron, :], self.dt.item()
        )

        # if binned:
        tr_t_spike = []
        for sp in spiketimes:
            tr_t_spike.append(
                self.ind_to_train(torch.tensor(sp), rate.shape[-1]).numpy()
            )

        return np.array(tr_t_spike).reshape(rate.shape[0], -1, rate.shape[-1])

        # else:
        #    return spiketimes
    def log_renewal_density(self, ISI):
        raise NotImplementedError
        
    def cum_renewal_density(self, ISI):
        raise NotImplementedError
    
    def log_survival(self, ISI):
        raise NotImplementedError

### classes ###
class FactorizedLikelihood(object):
    """
    The likelihood model class, p(yₙ|fₙ)

    variational_expectation() computes all E_q(f) related quantities and its derivatives for NGD
    We allow multiple f (vector fₙ) to correspond to a single yₙ as in heteroscedastic likelihoods

    The default functions here use cubature/MC approximation methods, exact integration is specific
    to certain likelihood classes.
    """

    def __init__(self, out_dims, f_dims, hyp):
        """
        all hyperparameters stored are pre-transformation
        :param hyp: (hyper)parameters of the likelihood model
        """
        self.hyp = hyp
        self.f_dims = f_dims
        self.out_dims = out_dims
        self.num_f_per_out = self.f_dims // self.out_dims

    @partial(jit, static_argnums=(0, 5))
    def grads_log_likelihood_n(self, f_mean, df_points, y, lik_params, derivatives):
        """
        Factorization over data points n, vmap over out_dims
        vmap over approx_points

        :param np.array f_mean: (scalar) or (cubature,) for cubature_dim > 1
        :param np.array df_points: (approx_points,) or (cubature, approx_points) for cubature_dim > 1

        :return:
            expected log likelihood ll (approx_points,)
            dll_dm (approx_points,) or (cubature, approx_points)
            d2ll_dm2 (approx_points,) or (cubature, cubature, approx_points)
        """
        f = (
            df_points + f_mean
        )  # (approx_points,) or (cubature, approx_points) for cubature_dim > 1

        if derivatives:

            def grad_func(f):
                ll, dll_dm = value_and_grad(self.log_likelihood_n, argnums=0)(
                    f, y, lik_params
                )
                return dll_dm, (ll, dll_dm)

            def temp_func(f):
                # dll_dm, (ll,) = grad_func(f)
                d2ll_dm2, aux = jacrev(grad_func, argnums=0, has_aux=True)(f)
                ll, dll_dm = aux
                return ll, dll_dm, d2ll_dm2

            ll, dll_dm, d2ll_dm2 = vmap(temp_func, in_axes=0, out_axes=(0, 0, 0))(f)

        else:
            ll = vmap(self.log_likelihood_n, (0, None, None))(f, y, lik_params)
            dll_dm, d2ll_dm2 = None, None

        return ll, dll_dm, d2ll_dm2

    def log_likelihood_n(self, f, y, lik_params):
        raise NotImplementedError(
            "direct evaluation of this log-likelihood is not implemented"
        )

    @staticmethod
    def link_fn(latent_mean):
        return latent_mean

    @partial(jit, static_argnums=(0, 8))
    def variational_expectation(
        self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=False
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

    ### evaluate ###
    def posterior_predictive(self, f_mean, f_cov):
        """
        Given posterior of q(f), compute q(a) for likelihood variables
        """
        raise NotImplementedError(
            "Posterior evaluation of likelihood variables not implemented"
        )

    def posterior_sample(self, f_samp):
        """
        Sample from posterior
        """
        raise NotImplementedError(
            "Posterior sampling of likelihood variables not implemented"
        )


class filtered_likelihood(base._likelihood):
    """
    Wrapper for base._likelihood classes with filters.
    """

    def __init__(self, likelihood, filter_obj):
        """ """
        if filter_obj.tensor_type != likelihood.tensor_type:
            raise ValueError("Filter and likelihood tensor types do not match")

        super().__init__(
            likelihood.tbin.item(),
            likelihood.F_dims,
            likelihood.neurons,
            likelihood.inv_link,
            likelihood.tensor_type,
            likelihood.mode,
        )
        self.add_module("likelihood", likelihood)
        self.add_module("filter", filter_obj)
        self.history_len = self.filter.filter_len - 1  # excludes instantaneous part

    def KL_prior(self, importance_weighted=False):
        return self.filter.KL_prior(importance_weighted) + self.likelihood.KL_prior(
            importance_weighted
        )

    def set_Y(self, spikes, batch_info):
        if len(spikes.shape) == 2:  # add in trial dimension
            spikes = spikes[None, ...]

        in_spikes = spikes[..., self.history_len :]
        self.likelihood.set_Y(
            in_spikes, batch_info
        )  # excludes history part of spike train
        self.likelihood.all_spikes = spikes.type(
            self.likelihood.tensor_type
        )  # overwrite

        _, batch_link, batch_initial = self.likelihood.batch_info
        if any(batch_initial[1:]) or all(batch_link[1:]) is False:
            raise ValueError("Filtered likelihood must take in continuous data")

        self.all_spikes = self.likelihood.all_spikes
        self.batch_info = self.likelihood.batch_info
        self.batches = self.likelihood.batches
        self.trials = self.likelihood.trials
        self.tsteps = self.likelihood.tsteps

    def constrain(self):
        """
        Constrain parameters in optimization
        """
        self.likelihood.constrain()
        self.filter.constrain()

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        spike coupling filter
        """
        batch_edge, _, _ = self.batch_info
        spk = self.all_spikes[
            :, neuron, batch_edge[b] : batch_edge[b + 1] + self.history_len
        ].to(self.likelihood.tbin.device)
        spk_filt, spk_var = self.filter(spk, XZ)  # trials, neurons, timesteps
        mean = F_mu + spk_filt
        variance = F_var + spk_var
        return self.likelihood.objective(mean, variance, XZ, b, neuron, samples, mode)

    def filtered_rate(self, F_mu, F_var, unobs_neuron, trials, MC_samples=1):
        """
        Evaluate the instantaneous rate after spike coupling, with unobserved neurons not contributing
        to the filtered population rate.
        """
        unobs_neuron = self.likelihood._validate_neuron(unobs_neuron)
        batch_edge, _, _ = self.batch_info
        spk = self.all_spikes[
            :, neuron, batch_edge[b] : batch_edge[b + 1] + self.history_len
        ].to(self.likelihood.tbin.device)

        with torch.no_grad():
            hist, hist_var = self.spike_filter(spk, XZ)
            hist[:, unobs_neuron, :] = 0  # mask
            hist_var[:, unobs_neuron, :] = 0  # mask
            h = self.mc_gen(
                F_mu + hist, F_var + hist_var, MC_samples, torch.arange(self.neurons)
            )
            intensity = self.likelihood.f(h.view(-1, trials, *h.shape[1:]))

        return intensity

    def sample(self, F_mu, ini_train, neuron=None, XZ=None, obs_spktrn=None):
        """
        Assumes all neurons outside neuron are observed for spike filtering.

        :param torch.Tensor F_mu: input F values of shape (MC, trials, neurons, time)
        :param np.ndarray ini_train: initial spike train of shape (trials, neurons, time)
        :returns: spike train and instantaneous firing rates of shape (trials, neurons, time)
        :rtype: tuple of np.array
        """
        neuron = self.likelihood._validate_neuron(neuron)
        n_ = list(range(self.likelihood.neurons))

        MC, trials, N, steps = F_mu.shape
        if trials != ini_train.shape[0] or N != ini_train.shape[1]:
            raise ValueError("Initial spike train shape must match input F tensor.")
        spikes = []
        spiketrain = torch.empty(
            (*ini_train.shape[:2], self.history_len), device=self.likelihood.tbin.device
        )

        iterator = tqdm(range(steps), leave=False)  # AR sampling
        rate = []
        for t in iterator:
            if t == 0:
                spiketrain[..., :-1] = torch.tensor(
                    ini_train, device=self.likelihood.tbin.device
                )
            else:
                spiketrain[..., :-2] = spiketrain[..., 1:-1].clone()  # shift in time
                spiketrain[..., -2] = torch.tensor(
                    spikes[-1], device=self.likelihood.tbin.device
                )

            with torch.no_grad():  # spiketrain last time element is dummy, [:-1] used
                if XZ is None:
                    cov_ = None
                else:
                    cov_ = XZ[:, t : t + self.history_len, :]
                hist, hist_var = self.filter(spiketrain, cov_)

            rate_ = (
                self.likelihood.f(F_mu[..., t] + hist[None, ..., 0])
                .mean(0)
                .cpu()
                .numpy()
            )  # (trials, neuron)

            if obs_spktrn is None:
                spikes.append(
                    self.likelihood.sample(rate_[..., None], n_, XZ=XZ)[..., 0]
                )
                # spikes.append(point_process.gen_IBP(1. - np.exp(-rate_*self.likelihood.tbin.item())))
            else:  # condition on observed spike train partially
                spikes.append(obs_spktrn[..., t])
                spikes[-1][:, neuron] = self.likelihood.sample(
                    rate_[..., None], neuron, XZ=XZ
                )[..., 0]
                # spikes[-1][:, neuron] = point_process.gen_IBP(1. - np.exp(-rate_[:, neuron]*self.likelihood.tbin.item()))
            rate.append(rate_)

        rate = np.stack(rate, axis=-1)  # trials, neurons, timesteps
        spktrain = np.transpose(
            np.array(spikes), (1, 2, 0)
        )  # trials, neurons, timesteps

        return spktrain, rate


# GLM filters
class _filter(nn.Module):
    """
    GLM coupling filter base class.
    """

    def __init__(self, filter_len, conv_groups, tensor_type):
        """
        Filter length includes instantaneous part
        """
        super().__init__()
        self.conv_groups = conv_groups
        self.tensor_type = tensor_type
        if filter_len <= 0:
            raise ValueError("Filter length must be bigger than zero")
        self.filter_len = filter_len

    def forward(self):
        """
        Return filter values.
        """
        raise NotImplementedError

    def KL_prior(self, importance_weighted):
        """
        Prior of the filter model.
        """
        return 0

    def constrain(self):
        return

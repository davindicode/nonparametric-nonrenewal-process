from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

from ..base import module
from ..filters.base import Filter
from ..GP.markovian import GaussianLTI


class FilterModule(module):
    """
    Spiketrain filter + GP with optional SSGP latent states
    """

    spikefilter: Union[Filter, None]

    def __init__(self, spikefilter, array_type):
        if spikefilter is not None:  # checks
            assert spikefilter.array_type == array_type
        super().__init__(array_type)
        self.spikefilter = spikefilter

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.spikefilter,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model

    def _spiketrain_filter(self, spktrain):
        """
        Apply the spike train filter
        """
        return filtered, KL

    def ELBO(self, prng_state, x, t, num_samps):
        raise NotImplementedError

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        spike coupling filter
        """
        batch_edge, _, _ = self.batch_info
        spk = self.all_spikes[
            :, neuron, batch_edge[b] : batch_edge[b + 1] + self.history_len
        ].to(self.likelihood.dt.device)
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
        ].to(self.likelihood.dt.device)

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
            (*ini_train.shape[:2], self.history_len), device=self.likelihood.dt.device
        )

        iterator = tqdm(range(steps), leave=False)  # AR sampling
        rate = []
        for t in iterator:
            if t == 0:
                spiketrain[..., :-1] = torch.tensor(
                    ini_train, device=self.likelihood.dt.device
                )
            else:
                spiketrain[..., :-2] = spiketrain[..., 1:-1].clone()  # shift in time
                spiketrain[..., -2] = torch.tensor(
                    spikes[-1], device=self.likelihood.dt.device
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
                # spikes.append(point_process.gen_IBP(1. - np.exp(-rate_*self.likelihood.dt.item())))
            else:  # condition on observed spike train partially
                spikes.append(obs_spktrn[..., t])
                spikes[-1][:, neuron] = self.likelihood.sample(
                    rate_[..., None], neuron, XZ=XZ
                )[..., 0]
                # spikes[-1][:, neuron] = point_process.gen_IBP(1. - np.exp(-rate_[:, neuron]*self.likelihood.dt.item()))
            rate.append(rate_)

        rate = np.stack(rate, axis=-1)  # trials, neurons, timesteps
        spktrain = np.transpose(
            np.array(spikes), (1, 2, 0)
        )  # trials, neurons, timesteps

        return spktrain, rate


class FilterGPLVM(FilterModule):
    """
    Spiketrain filter + GP with optional SSGP latent states
    """

    ssgp: Union[GaussianLTI, None]

    def __init__(self, ssgp, spikefilter, array_type):
        if ssgp is not None:  # checks
            assert ssgp.array_type == array_type

        super().__init__(spikefilter, array_type)
        self.ssgp = ssgp

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.ssgp,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model

    def _sample_input_trajectories(self, x, t, num_samps):
        """
        Combines observed inputs with latent trajectories
        """
        if self.ssgp is not None:
            x_samples, KL_ss = self.ssgp.sample_posterior(
                ss_params,
                ss_var_params,
                prng_keys[0],
                num_samps,
                timedata,
                None,
                jitter,
                compute_KL=True,
            )  # (time, tr, x_dims, 1)

        return inputs, KL

    def _sample_input_marginals(self, x, t, num_samps):
        """
        Combines observed inputs with latent marginal samples
        """
        if self.ssgp is not None:  # filtering-smoothing
            x_samples, KL_ss = self.ssgp.evaluate_posterior(
                ss_params,
                ss_var_params,
                prng_keys[0],
                num_samps,
                timedata,
                None,
                jitter,
                compute_KL=True,
            )  # (time, tr, x_dims, 1)

        return inputs, KL


class FilterSwitchingSSGP(FilterModule):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    def __init__(self, switchgp, spikefilter, array_type):
        if switchgp is not None:  # checks
            assert switchgp.array_type == array_type
        super().__init__(array_type)
        self.switchgp = switchgp


class FilterDTGPSSM(FilterModule):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    def __init__(self, gpssm, spikefilter):
        if gpssm is not None:  # checks
            assert gpssm.array_type == array_type
        super().__init__(spikefilter.array_type)
        self.gpssm = gpssm

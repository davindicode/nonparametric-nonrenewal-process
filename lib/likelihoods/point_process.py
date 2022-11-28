import numbers

import numpy as np
import scipy.special as scsps
import scipy.stats as scstats

import torch
from torch.distributions import Bernoulli
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

from .. import base


# ISI distribution classes
class ISI_base:
    """
    Base class for ISI distributions in statistical goodness-of-fit tests.
    Takes in time-rescaled ISIs.
    """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def prob_dens(self, tau):
        return self.rv.pdf((tau - self.loc) / self.scale) / self.scale

    def cum_prob(self, tau, loc=0.0, scale=1.0):
        return self.rv.cdf((tau - self.loc) / self.scale)

    def sample(self, shape):
        return self.rv.rvs(size=shape + self.event_shape) * self.scale + self.loc

    def intensity(self, tau, prev_tau):
        """
        Evaluates :math:'\lambda(\tau|\tau')'
        """
        return prob_dens(tau) / (1.0 - cum_prob(tau - prev_tau))


class ISI_gamma(ISI_base):
    r"""
    Gamma distribution

    ..math::

    """

    def __init__(self, shape, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.rv = scstats.gamma(shape)
        if isinstance(shape, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = shape.shape


class ISI_invgamma(ISI_base):
    r"""
    Inverse Gamma distribution

    ..math::

    """

    def __init__(self, shape, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.rv = scstats.invgamma(shape)
        if isinstance(shape, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = shape.shape


class ISI_invGauss(ISI_base):
    r"""
    Inverse Gaussian distribution
    Gives parameterization on wikipedia with lambda = 1

    ..math::

    """

    def __init__(self, mu, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.rv = scstats.invgauss(mu)
        if isinstance(mu, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = mu.shape


class ISI_logNormal(ISI_base):
    r"""
    Log normal renewal distribution
    Gives parameterization on wikipedia, mu = 0

    ..math::

    """

    def __init__(self, sigma, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.rv = scstats.lognorm(sigma)
        if isinstance(sigma, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = sigma.shape


### sampling ###
def gen_IRP(interval_dist, rate, tbin, samples=100):
    """
    Sample event times from an Inhomogenous Renewal Process with a given rate function
    samples is an algorithm parameter, should be around the expect number of spikes
    Assumes piecewise constant rate function

    Samples intervals from :math:`q(\Delta)`, parallelizes sampling

    :param np.ndarray rate: (trials, neurons, timestep)
    :param ISI_dist interval_dist: renewal interval distribution :math:`q(\tau)`
    :returns: event times as integer indices of the rate array time dimension
    :rtype: list of spike time indices as indexed by rate time dimension
    """
    sim_samples = rate.shape[2]
    N = rate.shape[1]  # neurons
    trials = rate.shape[0]
    T = (
        np.transpose(np.cumsum(rate, axis=-1), (2, 0, 1)) * tbin
    )  # actual time to rescaled, (time, trials, neuron)

    psT = 0
    sT_cont = []
    while True:

        sT = psT + np.cumsum(
            interval_dist.sample(
                (
                    samples,
                    trials,
                )
            ),
            axis=0,
        )
        sT_cont.append(sT)

        if not (T[-1, ...] >= sT[-1, ...]).any():  # all False
            break

        psT = np.tile(sT[-1:, ...], (samples, 1, 1))

    sT_cont = np.stack(sT_cont, axis=0).reshape(-1, trials, N)
    samples_tot = sT_cont.shape[0]
    st = []

    iterator = tqdm(range(samples_tot), leave=False)
    for ss in iterator:  # AR assignment
        comp = np.tile(sT_cont[ss : ss + 1, ...], (sim_samples, 1, 1))
        st.append(np.argmax((comp < T), axis=0))  # convert to rescaled time indices

    st = np.array(st)  # (samples_tot, trials, neurons)
    st_new = []
    for st_ in st.reshape(samples_tot, -1).T:
        if not st_.any():  # all zero
            st_new.append(np.array([]).astype(int))
        else:  # first zeros in this case counts as a spike indices
            for k in range(samples_tot):
                if st_[-1 - k] != 0:
                    break
            if k == 0:
                st_new.append(st_)
            else:
                st_new.append(st_[:-k])

    return st_new  # list of len trials x neurons


def gen_IBP(intensity):
    """
    Sample of the Inhomogenous Bernoulli Process

    :param numpy.array intensity: intensity of the Bernoulli process at array index
    :returns: sample of binary variables with same shape as intensity
    :rtype: numpy.array
    """
    b = Bernoulli(torch.tensor(intensity))
    return b.sample().numpy()


def gen_CMP(mu, nu, max_rejections=1000):
    """
    Use rejection sampling to sample from the COM-Poisson count distribution. [1]

    References:

    [1] `Bayesian Inference, Model Selection and Likelihood Estimation using Fast Rejection
         Sampling: The Conway-Maxwell-Poisson Distribution`, Alan Benson, Nial Friel (2021)

    :param numpy.array rate: input rate of shape (..., time)
    :param float tbin: time bin size
    :param float eps: order of magnitude of P(N>1)/P(N<2) per dilated Bernoulli bin
    :param int max_count: maximum number of spike counts per bin possible
    :returns: inhomogeneous Poisson process sample
    :rtype: numpy.array
    """
    trials = mu.shape[0]
    neurons = mu.shape[1]
    Y = np.empty(mu.shape)

    for tr in range(trials):
        for n in range(neurons):
            mu_, nu_ = mu[tr, n, :], nu[tr, n, :]

            # Poisson
            k = 0
            left_bins = np.where(nu_ >= 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                y_dash = torch.poisson(torch.tensor(mu__)).numpy()
                _mu_ = np.floor(mu__)
                alpha = (
                    mu__ ** (y_dash - _mu_)
                    / scsps.factorial(y_dash)
                    * scsps.factorial(_mu_)
                ) ** (nu__ - 1)

                u = np.random.rand(*mu__.shape)
                selected = u <= alpha
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError("Maximum rejection steps exceeded")
                else:
                    k += 1

            # geometric
            k = 0
            left_bins = np.where(nu_ < 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                p = 2 * nu__ / (2 * mu__ * nu__ + 1 + nu__)
                u_0 = np.random.rand(*p.shape)

                y_dash = np.floor(np.log(u_0) / np.log(1 - p))
                a = np.floor(mu__ / (1 - p) ** (1 / nu__))
                alpha = (1 - p) ** (a - y_dash) * (
                    mu__ ** (y_dash - a) / scsps.factorial(y_dash) * scsps.factorial(a)
                ) ** nu__

                u = np.random.rand(*mu__.shape)
                selected = u <= alpha
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError("Maximum rejection steps exceeded")
                else:
                    k += 1

    return Y

    """
    if rate.max() == 0:
        return np.zeros_like(rate)
        
    dt_ = np.sqrt(eps)/rate.max()
    dilation = max(int(np.ceil(tbin/dt_)), 1) # number of counts to allow per original bin
    if dilation > max_count:
        raise ValueError('Maximum count ({}, requested {}) exceeded for Poisson process sampling'.format(max_count, dilation))
    tbin_ = tbin / dilation
    rate_ = np.repeat(rate, dilation, axis=-1) # repeat to allow IBP to sum to counts > 1
    return gen_IBP(rate_*tbin_).reshape(*rate.shape[:-1], -1, dilation).sum(-1)
    """


# Poisson point process
class Poisson_pp(base._likelihood):
    def __init__(
        self, tbin, neurons, inv_link, tensor_type=torch.float, allow_duplicate=True
    ):
        super().__init__(tbin, neurons, neurons, inv_link, tensor_type)
        self.allow_duplicate = allow_duplicate

    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)

        tfact is the log of time_bin times the spike count
        """
        if self.allow_duplicate is False and spikes.max() > 1:  # only binary trains
            raise ValueError("Only binary spike trains are accepted in set_Y() here")
        super().set_Y(spikes, batch_info)

    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for sample evaluation. Note that spikes is batched including history
        when the model uses history couplings, hence we sample the spike batches without the
        history segments from this function.
        """
        rates = self.f(h)  # watch out for underflow or overflow here
        batch_edge, _, _ = self.batch_info
        spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
            self.tbin.device
        )
        if (
            self.trials != 1 and samples > 1 and self.trials < h.shape[0]
        ):  # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1)  # MC x trials

        if self.inv_link == "exp":  # spike count times log rate
            n_l_rates = spikes * h
        else:
            n_l_rates = spikes * torch.log(rates + 1e-12)

        return rates, n_l_rates, spikes

    def nll(self, b, rates, n_l_rates):
        nll = -n_l_rates + rates * self.tbin
        return nll.sum(1)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This
        is all combined in to the same dimension to be summed over. The weights :math:`w_s`
        are the quadrature weights or equal weights for MC, with appropriate normalization.

        :param int samples: number of MC samples or GH points (exact will ignore and give 1)

        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if mode == "MC":
            h = self.mc_gen(F_mu, F_var, samples, neuron)  # h has only observed neurons
            rates, n_l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = torch.tensor(1.0 / rates.shape[0])
        elif mode == "GH":
            h, ws = self.gh_gen(F_mu, F_var, samples, neuron)
            rates, n_l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
        elif mode == "direct":
            rates, n_l_rates, spikes = self.sample_helper(
                F_mu[:, neuron, :], b, neuron, samples
            )
            ws = torch.tensor(1.0 / rates.shape[0])
        else:
            raise NotImplementedError

        return self.nll(b, rates, n_l_rates), ws

    def sample(self, rate, neuron=None, XZ=None):
        """
        Approximate by a Bernoulli process, slight bias introduced as spike means at least one spike

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        return gen_IBP(rate[:, neuron, :] * self.tbin.item())


# renewal distributions
class _renewal_model(base._likelihood):
    """
    Renewal model base class
    """

    def __init__(
        self, tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize
    ):
        super().__init__(tbin, neurons, neurons, inv_link, tensor_type)
        self.allow_duplicate = allow_duplicate
        self.dequant = dequantize

    def train_to_ind(self, train):
        if self.allow_duplicate:
            duplicate = False
            spike_ind = train.nonzero().flatten()
            bigger = torch.where(train > 1)[0]
            add_on = (spike_ind,)
            for b in bigger:
                add_on += (
                    b * torch.ones(int(train[b]) - 1, device=train.device, dtype=int),
                )

            if len(add_on) > 1:
                duplicate = True
            spike_ind = torch.cat(add_on)
            return torch.sort(spike_ind)[0], duplicate
        else:
            return torch.nonzero(train).flatten(), False

    def ind_to_train(self, ind, timesteps):
        train = torch.zeros((timesteps))
        train[ind] += 1
        return train

    def rate_rescale(self, neuron, spike_ind, rates, duplicate, minimum=1e-8):
        """
        Rate rescaling with option to dequantize, which will be random per sample.

        :param torch.Tensor rates: input rates of shape (trials, neurons, timesteps)
        :returns: list of rescaled ISIs, list index over neurons, elements of shape (trials, ISIs)
        :rtype: list
        """
        rtime = torch.cumsum(rates, dim=-1) * self.tbin
        samples = rtime.shape[0]
        rISI = []
        for tr in range(self.trials):
            isis = []
            for en, n in enumerate(neuron):
                if len(spike_ind[tr][n]) > 1:
                    if self.dequant:
                        deqn = (
                            torch.rand(
                                samples, *spike_ind[tr][n].shape, device=rates.device
                            )
                            * rates[tr :: self.trials, en, spike_ind[tr][n]]
                            * self.tbin
                        )  # assume spike at 0

                        tau = rtime[tr :: self.trials, en, spike_ind[tr][n]] - deqn

                        if duplicate[tr, n]:  # re-oder in case of duplicate spike_ind
                            tau = torch.sort(tau, dim=-1)[0]

                    else:
                        tau = rtime[tr :: self.trials, en, spike_ind[tr][n]]

                    a = tau[:, 1:] - tau[:, :-1]
                    a[a < minimum] = minimum  # don't allow near zero ISI
                    isis.append(a)  # samples, order
                else:
                    isis.append([])
            rISI.append(isis)

        return rISI

    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.act [neuron_dim, batch_dim]
        """
        if self.allow_duplicate is False and spikes.max() > 1:  # only binary trains
            raise ValueError("Only binary spike trains are accepted in set_Y() here")
        super().set_Y(spikes, batch_info)
        batch_edge, _, _ = self.batch_info

        self.spiketimes = []
        self.intervals = torch.empty((self.batches, self.trials, self.neurons))
        self.duplicate = np.empty((self.batches, self.trials, self.neurons), dtype=bool)
        for b in range(self.batches):
            spk = self.all_spikes[..., batch_edge[b] : batch_edge[b + 1]]
            spiketimes = []
            for tr in range(self.trials):
                cont = []
                for k in range(self.neurons):
                    s, self.duplicate[b, tr, k] = self.train_to_ind(spk[tr, k])
                    cont.append(s)
                    self.intervals[b, tr, k] = len(s) - 1
                spiketimes.append(cont)
            self.spiketimes.append(
                spiketimes
            )  # batch list of trial list of spike times list over neurons

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
            self.tbin.device
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

        spiketimes = [[s.to(self.tbin.device) for s in ss] for ss in self.spiketimes[b]]
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
            self.ISI_dist(neuron), rate[:, neuron, :], self.tbin.item()
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


class Gamma(_renewal_model):
    """
    Gamma renewal process
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        shape,
        tensor_type=torch.float,
        allow_duplicate=True,
        dequantize=True,
    ):
        """
        Renewal parameters shape can be shared for all neurons or independent.
        """
        super().__init__(
            tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize
        )
        self.register_parameter("shape", Parameter(shape.type(self.tensor_type)))

    def constrain(self):
        self.shape.data = torch.clamp(self.shape.data, min=1e-5, max=2.5)

    def nll(self, n_l_rates, rISI, neuron):
        """
        Gamma case, approximates the spiketrain NLL (takes tbin into account for NLL).

        :param np.ndarray neuron: fit over given neurons, must be an array
        :param torch.Tensor F_mu: F_mu product with shape (samples, neurons, timesteps)
        :param torch.Tensor F_var: variance of the F_mu values, same shape
        :param int b: batch number
        :param np.ndarray neuron: neuron indices that are used
        :param int samples: number of MC samples for likelihood evaluation
        :returns: NLL array over sample dimensions
        :rtype: torch.tensor
        """
        samples_ = n_l_rates.shape[
            0
        ]  # ll_samplesxcov_samples, in case of trials trial_num=cov_samples
        shape_ = self.shape.expand(1, self.F_dims)[:, neuron]

        # Ignore the end points of the spike train
        # d_Lambda_i = rates[:self.spiketimes[0]].sum()*self.tbin
        # d_Lambda_f = rates[self.spiketimes[ii]:].sum()*self.tbin
        # l_start = torch.empty((len(neuron)), device=self.tbin.device)
        # l_end = torch.empty((len(neuron)), device=self.tbin.device)
        # l_start[n_enu] = torch.log(sps.gammaincc(self.shape.item(), d_Lambda_i))
        # l_end[n_enu] = torch.log(sps.gammaincc(self.shape.item(), d_Lambda_f))

        intervals = torch.zeros((samples_, len(neuron)), device=self.tbin.device)
        T = torch.empty(
            (samples_, len(neuron)), device=self.tbin.device
        )  # MC samples, neurons
        l_Lambda = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        for tr, isis in enumerate(rISI):  # over trials
            for n_enu, isi in enumerate(isis):  # over neurons
                if len(isi) > 0:  # nonzero number of ISIs
                    intervals[tr :: self.trials, n_enu] = isi.shape[-1]
                    T[tr :: self.trials, n_enu] = isi.sum(-1)
                    l_Lambda[tr :: self.trials, n_enu] = torch.log(isi + 1e-12).sum(-1)

                else:
                    T[
                        tr :: self.trials, n_enu
                    ] = 0  # TODO: minibatching ISIs approximate due to b.c.
                    l_Lambda[tr :: self.trials, n_enu] = 0

        nll = (
            -(shape_ - 1) * l_Lambda
            - n_l_rates
            + T
            + intervals[None, :] * torch.lgamma(shape_)
        )
        return nll.sum(1, keepdims=True)  # sum over neurons, keep as dummy time index

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        return super().objective(
            F_mu, F_var, XZ, b, neuron, self.shape, samples=samples, mode=mode
        )

    def ISI_dist(self, n):
        shape = self.shape[n].data.cpu().numpy()
        return ISI_gamma(shape, scale=1.0 / shape)


class log_Normal(_renewal_model):
    """
    Log-normal ISI distribution
    Ignores the end points of the spike train in each batch
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        sigma,
        tensor_type=torch.float,
        allow_duplicate=True,
        dequantize=True,
    ):
        """
        :param np.ndarray sigma: :math:`$sigma$` parameter which is > 0
        """
        super().__init__(
            tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize
        )
        # self.register_parameter('mu', Parameter(torch.tensor(mu, dtype=self.tensor_type)))
        self.register_parameter("sigma", Parameter(sigma.type(self.tensor_type)))
        self.register_buffer(
            "twopi_fact", 0.5 * torch.tensor(2 * np.pi, dtype=self.tensor_type).log()
        )

    def constrain(self):
        self.sigma.data = torch.clamp(self.sigma.data, min=1e-5)

    def set_Y(self, spikes, batch_info):
        super().set_Y(spikes, batch_info)

    def nll(self, n_l_rates, rISI, neuron):
        """
        Computes the log Normal distribution

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).

        :param torch.Tensor n_l_rates: log rates at spike times (samples, neurons, timesteps)
        :param torch.Tensor rISI: modified rate rescaled ISIs
        :param np.ndarray neuron: neuron indices that are used
        :returns: spike train negative log likelihood of shape (timesteps, samples (dummy dimension))
        :rtype: torch.tensor
        """
        sigma_ = self.sigma.expand(1, self.F_dims)[:, neuron]
        samples_ = n_l_rates.shape[0]

        l_Lambda = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        quad_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        norm_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        for tr, isis in enumerate(rISI):
            for n_enu, isi in enumerate(isis):
                if len(isi) > 0:  # nonzero number of ISIs
                    intervals = isi.shape[1]
                    l_Lambda[tr :: self.trials, n_enu] = torch.log(isi + 1e-12).sum(-1)
                    quad_term[tr :: self.trials, n_enu] = 0.5 * (
                        (torch.log(isi + 1e-12) / sigma_[:, n_enu : n_enu + 1]) ** 2
                    ).sum(
                        -1
                    )  # -mu_[:, n_enu:n_enu+1]
                    norm_term[tr :: self.trials, n_enu] = intervals * (
                        torch.log(sigma_[0, n_enu]) + self.twopi_fact
                    )

                else:
                    l_Lambda[tr :: self.trials, n_enu] = 0
                    quad_term[tr :: self.trials, n_enu] = 0
                    norm_term[tr :: self.trials, n_enu] = 0

        nll = -n_l_rates + norm_term + l_Lambda + quad_term
        return nll.sum(1, keepdims=True)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        return super().objective(
            F_mu,
            F_var,
            XZ,
            b,
            neuron,
            torch.exp(-self.sigma**2 / 2.0),
            samples=samples,
            mode=mode,
        )

    def ISI_dist(self, n):
        sigma = self.sigma[n].data.cpu().numpy()
        return ISI_logNormal(sigma, scale=np.exp(sigma**2 / 2.0))


class inv_Gaussian(_renewal_model):
    """
    Inverse Gaussian ISI distribution
    Ignores the end points of the spike train in each batch
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        mu,
        tensor_type=torch.float,
        allow_duplicate=True,
        dequantize=True,
    ):
        """
        :param np.ndarray mu: :math:`$mu$` parameter which is > 0
        """
        super().__init__(
            tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize
        )
        self.register_parameter("mu", Parameter(mu.type(self.tensor_type)))
        # self.register_parameter('lambd', Parameter(torch.tensor(lambd, dtype=self.tensor_type)))
        self.register_buffer(
            "twopi_fact", 0.5 * torch.tensor(2 * np.pi, dtype=self.tensor_type).log()
        )

    def constrain(self):
        self.mu.data = torch.clamp(self.mu.data, min=1e-5)

    def set_Y(self, spikes, batch_info):
        super().set_Y(spikes, batch_info)

    def nll(self, n_l_rates, rISI, neuron):
        """
        :param torch.Tensor F_mu: F_mu product with shape (samples, neurons, timesteps)
        :param torch.Tensor F_var: variance of the F_mu values, same shape
        :param int b: batch number
        :param np.ndarray neuron: neuron indices that are used
        :param int samples: number of MC samples for likelihood evaluation
        """
        mu_ = self.mu.expand(1, self.F_dims)[:, neuron]
        samples_ = n_l_rates.shape[0]

        l_Lambda = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        quad_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        norm_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        for tr, isis in enumerate(rISI):
            for n_enu, isi in enumerate(isis):
                if len(isi) > 0:  # nonzero number of ISIs
                    intervals = isi.shape[1]
                    l_Lambda[tr :: self.trials, n_enu] = torch.log(isi + 1e-12).sum(-1)
                    quad_term[tr :: self.trials, n_enu] = 0.5 * (
                        ((isi - mu_[:, n_enu : n_enu + 1]) / mu_[:, n_enu : n_enu + 1])
                        ** 2
                        / isi
                    ).sum(
                        -1
                    )  # (lambd_[:, n_enu:n_enu+1])
                    norm_term[tr :: self.trials, n_enu] = intervals * (
                        self.twopi_fact
                    )  # - 0.5*torch.log(lambd_[0, n_enu])

                else:
                    l_Lambda[tr :: self.trials, n_enu] = 0
                    quad_term[tr :: self.trials, n_enu] = 0
                    norm_term[tr :: self.trials, n_enu] = 0

        nll = -n_l_rates + norm_term + 1.5 * l_Lambda + quad_term
        return nll.sum(1, keepdims=True)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        return super().objective(
            F_mu, F_var, XZ, b, neuron, 1.0 / self.mu, samples=samples, mode=mode
        )

    def ISI_dist(self, n):
        """
        Note the scale parameter here is the inverse of the scale parameter in nll(), as the scale
        parameter here is :math:`\tau/s` while in nll() is refers to :math:`d\tau = s*r(t) \, \mathrm{d}t`
        """
        # self.lambd[n].data.cpu().numpy()
        mu = self.mu[n].data.cpu().numpy()
        return ISI_invGauss(mu, scale=mu)

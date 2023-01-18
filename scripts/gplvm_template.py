import argparse
import os

from functools import partial

import pickle

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from tqdm.autonotebook import tqdm


import sys

sys.path.append("..")
import lib

from lib.base import module
from lib.inference.base import Observations
from lib.inference.gp import ModulatedFactorized, RateRescaledRenewal, ModulatedRenewal, NonparametricPointProcess
from lib.inference.timeseries import GaussianLatentObservedSeries


### script ###
def standard_parser(parser):
    """
    Parser arguments belonging to training loop
    """
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoint/", action="store", type=str
    )

    parser.add_argument("--force_cpu", dest="force_cpu", action="store_true")
    parser.set_defaults(force_cpu=False)
    parser.add_argument("--double_arrays", dest="double_arrays", action="store_true")
    parser.set_defaults(double_arrays=False)
    parser.add_argument("--device", default=0, type=int)

    parser.add_argument("--seeds", default=[123], nargs="+", type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cov_MC", default=1, type=int)
    parser.add_argument("--ll_MC", default=10, type=int)

    parser.add_argument("--jitter", default=1e-6, type=float)
    parser.add_argument("--batch_size", default=10000, type=int)

    parser.add_argument("--lr_start", default=1e-2, type=float)
    parser.add_argument("--lr_end", default=1e-4, type=float)
    parser.add_argument("--lr_decay", default=0.98, type=float)
    parser.add_argument("--fix_param_names", default=[], nargs="+", type=str)
    
    parser.add_argument("--max_epochs", default=3000, type=int)
    parser.add_argument("--loss_margin", default=-1e0, type=float)
    parser.add_argument("--margin_epochs", default=100, type=int)

    parser.add_argument("--likelihood", action="store", type=str)
    parser.add_argument("--filter_type", default="", action="store", type=str)
    parser.add_argument("--observations", action="store", type=str)
    parser.add_argument("--observed_covs", default="", action="store", type=str)
    parser.add_argument("--latent_covs", default="", action="store", type=str)
    return parser



### inputs ###
def select_inputs(dataset_dict, observed_covs, likelihood):
    """
    Create the inputs to the model
    """
    
    ### ISIs ###
    if likelihood[:3] == 'isi':
        ISI_order = int(likelihood[3:])
        ISIs = dataset_dict["ISIs"][..., :ISI_order]
        
    else:
        ISIs = None
    
    ### observed covariates ###
    observed_covs_comps = observed_covs.split("-")
    covariates = dataset_dict["covariates"]
    
    input_data = []
    for xc in observed_covs_comps:
        if xc == "":
            continue
        input_data.append(covariates[xc])

    covs = np.stack(input_data, axis=-1)  # (ts, x_dims)
    return ISIs, covs



### kernel ###
def covariate_kernel(kernel_dicts, array_type):
    """
    Helper function for creating kernel triplet tuple
    """
    track_dims = 0
    dims_list, kernel_list = [], []
    for k, k_dict in enumerate(kernel_dicts):

        kernel_type = k_dict["type"]
        var_f = k_dict["var"]  # kernel variance (f_dims,)
        
        act = []
        for _ in lengthscales:
            act += [track_dims]
            track_dims += 1
        dims_list.append(act)

        if kernel_type == "linear":
            krn = lib.GP.kernels.Linear(f_dims)

        elif kernel_type == "polynomial":
            degree = k_dict["degree"]
            krn = lib.GP.kernels.Polynomial(
                bias=lengthscales,
                degree=degree,
                track_dims=act,
                f=kern_f,
                array_type=array_type,
            )
        
        else:  # lengthscale
            len_fx = k_dict["len"]  # kernel lengthscale (f_dims, x_dims)
            
            if kernel_type == "SE":
                kern = lib.GP.kernels.SquaredExponential(
                    f_dims, var_f, len_fx, metric_type='Euclidean', array_type=array_type)

            elif kernel_type == "circSE":
                kern = lib.GP.kernels.SquaredExponential(
                    f_dims, var_f, len_fx, metric_type='Cosine', array_type=array_type)

            elif kernel_type == "RQ":
                scale_mixture = k_dict["scale"]
                krn = kernels.RationalQuadratic(
                    f_dims, var_f, len_fx, scale_mixture, array_type=array_type)

            elif kernel_type == "Matern12":
                krn = lib.GP.kernels.Matern12(
                    f_dims, var_f, len_fx, array_type=array_type,
                )

            elif kernel_type == "Matern32":
                krn = lib.GP.kernels.Matern32(
                    f_dims, var_f, len_fx, array_type=array_type,
                )

            elif kernel_type == "Matern52":
                krn = lib.GP.kernels.Matern52(
                    f_dims, var_f, len_fx, array_type=array_type,
                )
                
            else:
                raise NotImplementedError("Kernel type is not supported.")

        kernel_list.append(krn)
        
    return kernel_list, dims_list



def ISI_kernel_dict_induc_list(isi_order, num_induc, out_dims):
    """
    Kernel for time since spike and ISI dimensions
    
    Note all ISIs are transformed to [0, 1] range, so lengthscales based on that
    """
    induc_list = []
    kernel_dicts = []

    ones = np.ones(out_dims)

    for _ in range(isi_order - 1):  # first time to spike handled by SSGP
        induc_list += [np.random.randn(num_induc)]
        kernel_tuples += [
            {"type": "SE", "var": ones, "len": np.ones((out_dims, 1))}]

    return kernel_dicts, induc_list


def latent_kernel_dict_induc_list(latent_covs, num_induc, out_dims):
    """
    Construct kernel tuples for latent space
    """
    latent_covs_comps = latent_covs.split("-")

    induc_list = []
    kernel_dicts = []

    for zc in latent_covs_comps:
        d_z = int(zc.split("d")[-1])
        induc_list += [np.random.randn(num_induc)] * d_z
        ls = np.ones((out_dims, d_z))
        var = np.ones(d_z)
        kernel_dicts += [{"type": "SE", "var": var, "len": ls}]

    return kernel_dicts, induc_list


def compactify_kernel_dicts(kernel_dicts, concat_params):
    """
    Stack together parameters for consecutive kernels of same type
    """
    compact_kernel_dicts = [kernel_dicts[0]]
    
    for kd in kernel_dicts[1:]:
        if kd["type"] == compact_kernel_dicts[-1]["type"]:
            for name in concat_params:
                compact_kernel_dicts[-1][names] = np.concatenate(
                    [compact_kernel_dicts[-1][name], kd[name]], axis=-1
                )
        else:
            compact_kernel_dicts += [kd]
    
    return compact_kernel_dicts



def build_kernel(observations, f_dims, isi_order, gen_obs_kernel_induc_func):
    """
    Assemble the kernel components
    """
    observations_comps = observations.split("-")
    num_induc = int(observations_comps[1])

    kernel_dicts, induc_list = [], []

    if observations_comps[0] == "nonparam_pp_gp":  # ISIs
        kernel_dicts_isi, induc_list_isi = ISI_kernel_dict_induc_list(isi_order, num_induc, f_dims)
        kernel_dicts += kernel_dicts_isi
        induc_list += induc_list_isi

    # observations
    obs_kernel_dicts, obs_induc_list = gen_obs_kernel_induc_func(observations, num_induc, f_dims)
    kernel_dicts += obs_kernel_dicts
    induc_list += obs_induc_list

    # latent
    kernel_dicts_lat, induc_list_lat = latent_kernel(latent_covs, num_induc, f_dims)
    kernel_dicts += kernel_dicts_lat
    induc_list += induc_list_lat

    # product kernel
    kernel_dicts = compactify_kernel_dicts(kernel_dicts, ["len"])
    kernel_list, dims_list = covariates_kernel_list(kernel_dicts, array_type)
    cov_kernel = lib.GP.kernels.Product(kernel_list, dims_list)
    induc_locs = np.array(induc_list)  # (f_dims, num_induc, x_dims)
    return cov_kernel, induc_locs



### observation models ###
def build_spikefilters(filter_type):                   
    """
    Create the spike coupling object.
    
    if spkcoupling_mode[:2] == 'gp':
        l0, l_b0, beta0, l_mean0, num_induc = filter_props
        if spkcoupling_mode[2:6] == 'full':
            D = neurons*neurons
        else:
            D = neurons
        l_t = l0*np.ones((1, D))
        l_b = l_b0*np.ones((1, D))
        beta = beta0*np.ones((1, D))
        filter_kernel = [('DSE', 'euclid', l_t, l_b, beta)]
        mean_func = nppl.likelihoods.filters.decaying_exponential(D, 0., l_mean0)
        filter_data = (hist_len, filter_kernel)
        
        #    if filter_data is not None: # add filter time dimension
        filter_len, filter_kernel = filter_data
        max_time = filter_len*tbin
        ind_list = [np.linspace(0, max_time, num_induc)] + ind_list
        temp = []
        for k in kernel_tuples_:
            temp.append(k[:1] + filter_kernel + k[1:])
        kernel_tuples_ = temp
        
    """
    if filter_type == "":
        return None
    filter_length = int(filter_type.split("H")[-1])
    
    if filter_type == 'sigmoid':
        alpha = np.ones((neurons, 1))
        beta = np.ones((neurons, 1))
        tau = 10*np.ones((neurons, 1))

        flt = lib.filters.FIR.SigmoidRefractory(
            alpha,
            beta,
            tau,
            filter_length,
        )
        
    elif filter_type[4:8] == 'svgp':
        num_induc = int(filter_type[8:])
        
        if filter_type[:4] == 'full':
            D = neurons*neurons
            out_dims = neurons
        elif filter_type[:4] == 'self':
            D = neurons
            out_dims = 1
            
        v = 100*tbin*torch.ones(1, D)
        l = 100.*tbin*torch.ones((1, D))
        l_b = 100.*tbin*torch.ones((1, D))
        beta = 1.*torch.ones((1, D))
        
        mean_func = nppl.kernels.means.decaying_exponential(D, 0., 100.*tbin)

        Xu = torch.linspace(0, hist_len*tbin, num_induc)[None, :, None].repeat(D, 1, 1)
        
        krn2 = nppl.kernels.kernel.DSE(
            input_dims=1, lengthscale=l, \
            lengthscale_beta=l_b, beta=beta, \
            track_dims=[0], f='exp', \
            tensor_type=tensor_type
        )
        kernelobj = nppl.kernels.kernel.Product(krn1, krn2)
        inducing_points = nppl.kernels.kernel.inducing_points(D, Xu, constraints=[], tensor_type=tensor_type)

        gp = nppl.mappings.SVGP(
            1, D, kernelobj, inducing_points=inducing_points, 
            whiten=True, jitter=1e-5, mean=mean_func, learn_mean=True
        )
        
        flt = lib.filters.GaussianProcess(out_dims, neurons, hist_len+1, tbin, gp, tens_type=tensor_type)
        
        
    elif filter_type[4:7] == 'rcb':
        strs = filter_type[7:].split('-')
        B, L, a, c = int(strs[0]), float(strs[1]), torch.tensor(float(strs[2])), torch.tensor(float(strs[3]))
        
        ini_var = 1.
        if filter_type[:4] == 'full':
            phi_h = np.linspace(0., L, B)[:, None, None].repeat(1, neurons, neurons)
            w_h = np.sqrt(ini_var) * np.random.randn(B, neurons, neurons)

        elif filter_type[:4] == 'self':
            phi_h = np.linspace(0., L, B)[:, None].repeat(1, neurons)
            w_h = np.sqrt(ini_var) * np.random.randn(B, neurons)
            
        a = np.ones((2, neurons, neurons))
        c = np.ones((2, neurons, neurons))
        w_h = np.random.randn(2, neurons, neurons)
        phi_h = np.ones((2, neurons, neurons))

        flt = lib.filters.FIR.RaisedCosineBumps(
            a,
            c,
            w_h,
            phi_h, 
            filter_length,
        )
        flt = lib.filters.RaisedCosineBumps(a=a, c=c, phi=phi_h, w=w_h, timesteps=hist_len+1, 
                                                                   learnable=[False, False, False, True], tensor_type=tensor_type)
        
    else:
        raise ValueError
    
    return flt



def build_factorized_gp(gen_obs_kernel_induc_func, observations, likelihood, tbin, obs_dims, obs_filter, jitter, array_type):
    """
    Build models with likelihoods that factorize over time steps
    """
    # likelihood
    likelihood_comps = likelihood.split("-")
    lik_type = likelihood_comps[0]
    
    if lik_type == "IBP":
        likelihood = lib.likelihoods.factorized.Bernoulli(
            obs_dims, tbin, likelihood_comps[-1], array_type=array_type
        )

    elif lik_type == "PP":
        likelihood = lib.likelihoods.PointProcess(
            obs_dims, tbin, likelihood_comps[-1], array_type=array_type
        )
        
    elif lik_type == "IP":
        likelihood = lib.likelihoods.Poisson(
            obs_dims, tbin, likelihood_comps[-1], array_type=array_type
        )

    elif lik_type == "ZIP":
        alpha = 0.1 * np.ones(obs_dims)
        likelihood = lib.likelihoods.ZeroInflatedPoisson(
            obs_dims, tbin, alpha, likelihood_comps[-1], array_type=array_type
        )

    elif lik_type == "hZIP":
        likelihood = lib.likelihoods.HeteroscedasticZeroInflatedPoisson(
            obs_dims, tbin, likelihood_comps[-1], array_type=array_type
        )

    elif lik_type == "NB":
        r_inv = 10.0 * np.ones(obs_dims)
        likelihood = lib.likelihoods.factorized.NegativeBinomial(
            obs_dims, tbin, r_inv, likelihood_comps[-1], array_type=array_type
        )

    elif lik_type == "hNB":
        likelihood = lib.likelihoods.factorized.HeteroscedasticNegativeBinomial(
            obs_dims, tbin, likelihood_comps[-1], array_type=array_type
        )

    elif lik_type == "CMP":
        J = int(likelihood_comps[1])
        log_nu = np.zeros(obs_dims)
        likelihood = lib.likelihoods.factorized.ConwayMaxwellPoisson(
            obs_dims, tbin, log_nu, likelihood_comps[-1], J=J, array_type=array_type
        )

    elif lik_type == "hCMP":
        J = int(likelihood_comps[1])
        likelihood = lib.likelihoods.factorized.HeteroscedasticConwayMaxwellPoisson(
            obs_dims, tbin, likelihood_comps[-1], J=J, array_type=array_type
        )

    elif lik_type == "U":
        gp_mean = None  # bias is already in the likelihood
        
        basis_mode = likelihood_comps[1]
        C = int(likelihood_comps[2])
        K = int(likelihood_comps[3])
        
        likelihood = lib.likelihoods.factorized.UniversalCount(
            obs_dims, C, K, basis_mode, array_type=array_type
        )

    else:
        raise NotImplementedError
    
    # kernel
    f_dims = likelihood.f_dims
    kernel, induc_locs = build_kernel(observations, f_dims, 0, gen_obs_kernel_induc_func)
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims
    
    # SVGP
    RFF_num_feats = int(observations.split("-")[2])
    num_induc = induc_locs.shape[1]
    u_mu = 1. + 0.*np.random.randn(prng_state, shape=(f_dims, num_induc, 1))
    u_Lcov = 0.01*np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)
    
    gp_mean = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(kernel, induc_locs, u_mu, u_Lcov, 
                               RFF_num_feats=RFF_num_feats, array_type=array_type)

    # model
    model = lib.inference.gp.ModulatedFactorized(svgp, gp_mean, likelihood, spikefilter=obs_filter)
    return model



def build_renewal_gp(gen_obs_kernel_induc_func, observations, renewal_type, dt, obs_dims, obs_filter, 
                     model_type, jitter, array_type):
    """
    Build models based on renewal densities
    """
    # likelihood
    if renewal_type == 'gamma':
        alpha = np.ones(obs_dims)
        renewal = lib.likelihoods.Gamma(
            obs_dims,
            dt,
            alpha,
        )

    elif renewal_type == 'lognorm':
        sigma = np.ones(obs_dims)
        renewal = lib.likelihoods.LogNormal(
            obs_dims,
            dt,
            sigma,
        )

    elif renewal_type == 'invgauss':
        mu = np.ones(obs_dims)
        renewal = lib.likelihoods.InverseGaussian(
            obs_dims,
            dt,
            mu,
        )
        
    else:
        raise ValueError('Invalid renewal likelihood')
        
    # kernel
    kernel, induc_locs = build_kernel(observations, obs_dims, 0, gen_obs_kernel_induc_func)
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims
    
    # SVGP
    RFF_num_feats = int(observations.split("-")[2])
    num_induc = induc_locs.shape[1]
    u_mu = 1. + 0.*np.random.randn(prng_state, shape=(f_dims, num_induc, 1))
    u_Lcov = 0.01*np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)
    
    gp_mean = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(kernel, induc_locs, u_mu, u_Lcov, 
                               RFF_num_feats=RFF_num_feats, array_type=array_type)

    Kzz = svgp.kernel.K(svgp.induc_locs, None, False)
    lambda_1, chol_Lambda_2 = lib.GP.sparse.t_from_q_svgp_moments(Kzz, u_mu, u_Lcov)

    # model
    if model_type == 'rescaled':
        model = lib.inference.gp.RateRescaledRenewal(svgp, gp_mean, renewal, spikefilter=flt)
        
    elif model_type == 'modulated':
        model = lib.inference.gp.ModulatedRenewal(svgp, gp_mean, renewal, spikefilter=flt)
        
    else:
        raise ValueError('Invalid renewal model type')
        
    return model



def build_nonparametric(gen_obs_kernel_induc_func, observations, likelihood, dt, obs_dims, 
                        ss_kernel, spatial_MF, fixed_grid_locs, 
                        jitter, array_type):
    """
    Build point process models with nonparametric CIFs
    
    :param jnp.ndarray site_locs: temporal inducing point locations (f_dims, num_induc)
    :param float dt: time bin size for spike train data
    """
    x_dims = ss_kernel.out_dims
#     if len(kernel_list) == 0:  # not modulated
#         Tsteps = 1000

#         site_locs = np.linspace(0., 1., Tsteps)[None, :].repeat(x_dims, axis=0)  # s
#         site_obs = 0. * np.ones([x_dims, Tsteps, 1]) + 0*np.random.randn(x_dims, Tsteps, 1)
#         site_Lcov = 1. * np.ones([x_dims, Tsteps, 1]) + 0*np.random.randn(x_dims, Tsteps, 1)

          # SSGP
#         gp = lib.GP.markovian.MultiOutputLTI(
#             ss_kernel, site_locs, site_obs, site_Lcov, fixed_grid_locs=True)
    
#     else:

    # kernels
    observations_comps = observations.split("-")
    ss_type = observations_comps[2]
    
    if ss_type == 'matern12':
        len_t = 1.0*np.ones((f_dims, 1))  # GP lengthscale
        var_t = 1.0*np.ones(f_dims)  # GP variance
        ss_kernel = lib.GP.kernels.Matern12(f_dims, variance=var_t, lengthscale=len_t)
        
    elif ss_type == 'matern32':
        len_t = 1.0*np.ones((f_dims, 1))  # GP lengthscale
        var_t = 1.0*np.ones(f_dims)  # GP variance
        ss_kernel = lib.GP.kernels.Matern32(f_dims, variance=var_t, lengthscale=len_t)
        
    elif ss_type == 'matern52':
        len_t = 1.0*np.ones((f_dims, 1))  # GP lengthscale
        var_t = 1.0*np.ones(f_dims)  # GP variance
        ss_kernel = lib.GP.kernels.Matern52(f_dims, variance=var_t, lengthscale=len_t)
        
    else:
        raise ValueError
    
    isi_order = int(likelihood[3:])
    kernel, induc_locs = build_kernel(observations, obs_dims, isi_order, gen_obs_kernel_induc_func)
    assert x_dims == kernel.in_dims
    f_dims = kernel.out_dims
    
    # spatiotemporal SVGP
    num_induc_t = int(observations_comps[3])
    num_induc_sp = induc_locs.shape[1]
    num_induc = num_induc_sp * num_induc_t
    
    site_locs = np.linspace(0., 1., num_induc_t)[None, :].repeat(f_dims, axis=0)
    site_obs = 0.1 * np.random.randn(f_dims, num_induc_t, num_induc_sp, 1)
    site_Lcov = 0.1 * np.eye(num_induc_sp)[None, None, ...].repeat(
        num_induc_t, axis=1).repeat(f_dims, axis=0)

    st_kernel = lib.GP.kernels.MarkovSparseKronecker(ss_kernel, kernel, induc_locs)
    
    RFF_num_feats = int(observations_comps[4])
    gp = lib.GP.spatiotemporal.KroneckerLTI(
        st_kernel, site_locs, site_obs, site_Lcov, spatial_MF, fixed_grid_locs, 
        RFF_num_feats=RFF_num_feats, array_type=array_type, 
    )
    
    # BNPP
    wrap_tau = 10.*np.ones((x_dims,))
    refract_tau = 1e0*np.ones((x_dims,))
    refract_neg= -12.
    mean_bias = 0.*np.ones((x_dims,))
    
    model = lib.inference.gp.NonparametricPointProcess(
        gp, wrap_tau, refract_tau, refract_neg, mean_bias, dt)
    return model


### model ###
class GPLVM(module):
    """
    base class for likelihoods
    """

    inp_model: GaussianLatentObservedSeries
    obs_model: Observations

    def __init__(self, inp_model, obs_model):
        """
        The logit link function:
        P = E[yₙ=1|fₙ] = 1 / 1 + exp(-fₙ)

        The Probit link function, i.e. the Error Function Likelihood:
        i.e. the Gaussian (Normal) cumulative density function:
        P = E[yₙ=1|fₙ] = Φ(fₙ)
                       = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙ],
        The Normal CDF is calulcated using the error function:
                       = (1 + erf(fₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
        """
        assert inp_model.array_type == obs_model.array_type
        super().__init__(obs_model.array_type)
        self.inp_model = inp_model
        self.obs_model = obs_model
        
    def ELBO(self, ts, xs, deltas, ys, ys_filt):
        if type(self.obs_model) == NonparametricPointProcess:
            xs, KL_x = self.inp_model.sample_marginal_posterior(xs)
            ELBO = self.obs_model.ELBO(ts, xs, deltas, ys)
            return ELBO + KL_x
        
        elif type(self.obs_model) == ModulatedFactorized:
            xs, KL_x = self.inp_model.sample_marginal_posterior(xs)
            ELBO = self.obs_model.ELBO(ts, xs, deltas, ys)
            return ELBO + KL_x
        
        elif type(self.obs_model) == RateRescaledRenewal:
            xs, KL_x = self.inp_model.sample_posterior(xs)
            ELBO = self.obs_model.ELBO(ts, xs, deltas, ys)
            return ELBO + KL_x
        
        else:
            raise ValueError



def setup_latents(d_x, latent_covs, site_lims, array_type):
    """
    latent covariates
    """
    latent_covs_comps = latent_covs.split("-")
    
    if len(latent_covs_comps) > 3:
        # settings
        num_site_locs = int(latent_covs_comps[-3])
        diagonal_site = (latent_covs_comps[-2] == 'diagonal')
        fixed_grid_locs = (latent_covs_comps[-1] == 'fixed_grid')
        site_locs = np.linspace(site_lims[0], site_lims[-1], num_site_locs)

        # list
        tot_d_z, ss_kernels = 0, []
        for zc in latent_covs_comps[:-3]:
            d_z = 0

            if zc[:9] == "matern12d":
                d_z = int(zc[9:])

                var_z = 1.0*np.ones((d_z))  # GP variance
                len_z = 1.0*np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(lib.GP.kernels.Matern12(d_z, variance=var_z, lengthscale=len_z))

            elif zc[:9] == "matern32d":
                d_z = int(zc[9:])

                var_z = 1.0*np.ones((d_z))  # GP variance
                len_z = 1.0*np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(lib.GP.kernels.Matern32(d_z, variance=var_z, lengthscale=len_z))

            elif zc[:9] == "matern52d":
                d_z = int(zc[9:])

                var_z = 1.0*np.ones((d_z))  # GP variance
                len_z = 1.0*np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(lib.GP.kernels.Matern52(d_z, variance=var_z, lengthscale=len_z))

            elif zc[:4] == "LEGd":
                d_z, d_s = [int(d) for d in latent_covs.split("s")]

                N = np.ones(d_s)[None]
                R = np.eye(d_s)[None]
                H = np.random.randn(d_z, d_s)[None] / np.sqrt(d_s)
                Lam = np.random.randn(d_s, d_s)[None] / np.sqrt(d_s)

                ss_kernels.append(lib.GP.kernels.LEG(N, R, H, Lam))

            elif zc != "":
                raise ValueError("Invalid latent covariate type")

            tot_d_z += d_z

            if len(ss_kernels) > 1:
                ss_kernels = lib.GP.kernels.StackMarkovian(ss_kernels)
            else:
                ss_kernels = ss_kernels[0]

            # site_init
            Tsteps = len(site_locs)
            site_obs = 0. * np.ones([Tsteps, tot_d_z, 1]) + 0*np.random.randn(Tsteps, tot_d_z, 1)
            site_Lcov = 1. * np.eye(tot_d_z)[None, ...].repeat(Tsteps, axis=0)

            ssgp = lib.GP.markovian.GaussianLTI(
                ss_kernels, site_locs, site_obs, site_Lcov, diagonal_site, fixed_grid_locs, array_type=array_type)

    else:
        tot_d_z, ssgp = 0, None
       
    # joint latent observed covariates
    lat_covs_dims = list(range(d_x, d_x + tot_d_z))
    obs_covs_dims = list(range(d_x))
    
    inputs_model = lib.inference.timeseries.GaussianLatentObservedSeries(
        ssgp, lat_covs_dims, obs_covs_dims, array_type=array_type)
    
    return inputs_model



def setup_observations(
    gen_obs_kernel_induc_func, likelihood, filter_type, observations, 
    observed_covs, latent_covs, obs_dims, tbin, jitter, array_type
):
    """
    Assemble the encoding model
    
    :return:
        used covariates, inputs model, observation model
    """
    ### GP observation model ###
    obs_filter = build_spikefilters(filter_type)
    observations_comps = observations.split("-")
    
    if observations_comps[0] == "factorized_gp":
        obs_model = build_factorized_gp(
            gen_obs_kernel_induc_func, observations, likelihood, tbin, obs_dims, obs_filter, 
            jitter, array_type, 
        )

    elif observations_comps[0] == "rate_renewal_gp":
        obs_model = build_renewal_gp(
            gen_obs_kernel_induc_func, observations, renewal_type, tbin, obs_dims, obs_filter, 
            "rescaled", jitter, array_type, 
        )
        
    elif observations_comps[0] == "mod_renewal_gp":
        obs_model = build_renewal_gp(
            gen_obs_kernel_induc_func, observations, renewal_type, tbin, obs_dims, obs_filter, 
            "modulated", jitter, array_type, 
        )
        
    elif observations_comps[0] == "nonparam_pp_gp":
        obs_model = build_nonparametric(
            gen_obs_kernel_induc_func, observations, likelihood, tbin, obs_dims, 
            ss_kernel, site_locs, spatial_MF, fixed_grid_locs, 
            jitter, array_type, 
        )
        
    else:
        raise ValueError('Invalid observation model type')
        
    return obs_model


def fit(parser_args, dataset_dict, observed_kernel_dict_induc_list, save_name):
    """
    Fit and save the model
    General training loop
    """
    config = argparse.Namespace(**{**vars(parser_args), **dataset_dict["properties"]})
    
    # JAX
    if config.force_cpu:
        jax.config.update("jax_platform_name", "cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.device}"

    if config.double_arrays:
        array_type = jnp.float64
        jax.config.update("jax_enable_x64", True)
    else:
        array_type = jnp.float32
    
    # data preparation
    ISIs, covariates = select_inputs(dataset_dict, config.observed_covs, config.likelihood)
    
    filter_length = int(config.filter_type.split("H")[-1]) if config.filter_type != "" else 0
    align_start_ind = dataset_dict["align_start_ind"]
    align_end_ind = align_start_ind + covariates.shape[0]
    ttslice = slice(align_start_ind-filter_length, align_end_ind)
    observations = dataset_dict["spiketrains"][:, ttslice]
    timestamps = dataset_dict["timestamps"]
    obs_dims, tbin = dataset_dict["properties"]["neurons"], dataset_dict["properties"]["tbin"]
    ISIs = ISIs.transpose(1, 0, 2) if ISIs is not None else None  # (ts, obs_dims, order)

    dataloader = lib.inference.timeseries.BatchedTimeSeries(
        timestamps, covariates, ISIs, observations, config.batch_size, filter_length)
    
    gen_kernel_induc_func = lambda observations, num_induc, out_dims: observed_kernel_dict_induc_list(
            observations, num_induc, out_dims, dataset_dict["covariates"])
    
    # optimizer
    learning_rate_schedule = optax.exponential_decay(
        init_value=config.lr_start,
        transition_steps=dataloader.batches,
        decay_rate=config.lr_decay,
        transition_begin=0,
        staircase=True,
        end_value=config.lr_end,
    )
    optim = optax.adam(learning_rate_schedule)

    # fitting
    seeds = config.seeds
    for seed in seeds:
        print("seed: {}".format(seed))

        # seed everything
        np.random.seed(seed)
        prng_state = jr.PRNGKey(seed)

        # create and initialize model
        obs_covs_dims = covariates.shape[-1]
        inp_model = setup_latents(
            obs_covs_dims, config.latent_covs, [timestamps[0], timestamps[-1]], array_type)
        
        obs_model = setup_observations(
            gen_kernel_induc_func, 
            config.likelihood, config.filter_type, 
            config.observations, config.observed_covs, 
            config.latent_covs, obs_dims, tbin, config.jitter, array_type, 
        )
        model = gplvm(inp_model, obs_model)

        # freeze parameters
        select_fixed_params = lambda tree: [
            getattr(tree, name) for name in config.fix_param_names
        ]

        filter_spec = jax.tree_map(lambda _: True, model)
        filter_spec = eqx.tree_at(
            select_learnable_params,
            filter_spec,
            replace=(False,) * len(fix_param_names),
        )
        
        # loss
        @partial(eqx.filter_value_and_grad, arg=filter_spec)
        def compute_loss(model, prng_state, data):
            ts, xs, deltas, ys, ys_filt = data
            nELBO = -model.ELBO(ts, xs, deltas, ys, ys_filt).mean()  # mean over MC
            return nELBO
        
        @partial(eqx.filter_jit, device=jax.devices()[0])
        def make_step(model, prng_state, data, opt_state):
            loss, grads = compute_loss(model, prng_state, data)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            model = model.apply_constraints(model)
            return loss, model, opt_state

        # initialize optimizers
        opt_state = optim.init(model)
        loss_tracker = []

        try:  # attempt to fit model
            iterator = tqdm(range(epochs))
            for ep in iterator:

                avg_loss = []
                for b in range(dataloader.batches):
                    batch_data = dataloader.load(b)
                    prng_state, _ = jr.split(prng_state)

                    loss, model, opt_state = make_step(model, prng_state, batch_data, opt_state)
                    loss = loss.item()
                    loss_tracker.append(loss)
                    avg_loss.append(loss)
                    
                    loss_dict = {"loss": loss}
                    iterator.set_postfix(**loss_dict)
                    
                abloss = np.array(avg_loss).mean()  # average over batches (subsampled estimator of loss)
                iterator.set_postfix(loss=sloss)
                tracked_loss.append(sloss)

                if sloss <= minloss + loss_margin:
                    cnt = 0
                else:
                    cnt += 1

                if sloss < minloss:
                    minloss = sloss

                if cnt > margin_epochs:
                    print("Stopped at epoch {}.".format(epoch + 1))
                    break

            # save and progress
            savefile = config.checkpoint_dir + save_name
            
            if os.path.exists(
                savefile + "_result.p"
            ):  # check previous best losses
                with open(savefile + "_result.p", "rb") as f:
                    results = pickle.load(f)
                    lowest_loss = results["training_loss"][-1]
            else:
                lowest_loss = np.inf  # nonconvex optimization, pick the best

            if losses[-1] < lowest_loss:
                if not os.path.exists(config.checkpoint_dir):
                    os.makedirs(config.checkpoint_dir)

                # save model
                pickle.dump(model, open(savefile + ".p", "wb"), pickle.HIGHEST_PROTOCOL)
                
                # save results
                savedata = {
                    "training_loss": losses,
                    "best_seed": seed,
                    "config": config,
                }
                pickle.dump(savedata, open(savefile + ".p", "wb"), pickle.HIGHEST_PROTOCOL)

        except (ValueError, RuntimeError) as e:
            print(e)

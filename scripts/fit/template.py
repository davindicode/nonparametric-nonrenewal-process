import argparse
import os

import pickle

import sys

from functools import partial, reduce

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from tqdm.autonotebook import tqdm

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

from gaussneuro.observations.base import Observations
from gaussneuro.observations.svgp import (
    ModulatedFactorized,
    ModulatedRenewal,
    RateRescaledRenewal,
)
from gaussneuro.observations.bnpp import NonparametricPointProcess
from gaussneuro.inputs.gaussian import GaussianLatentObservedSeries
from gaussneuro.models import GaussianTwoLayer
from gaussneuro.utils.loaders import BatchedTimeSeries


### script ###
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        a = attr.split('[')
        if len(a) > 1:  # list
            return getattr(obj, a[0], *args)[int(a[1].split(']')[0])]
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


def standard_parser(parser):
    """
    Parser arguments belonging to training loop
    """
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "--checkpoint_dir", default="../checkpoint/", action="store", type=str
    )

    parser.add_argument("--force_cpu", dest="force_cpu", action="store_true")
    parser.set_defaults(force_cpu=False)
    parser.add_argument("--double_arrays", dest="double_arrays", action="store_true")
    parser.set_defaults(double_arrays=False)
    parser.add_argument("--device", default=0, type=int)

    parser.add_argument("--seeds", default=[123], nargs="+", type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--num_MC", default=1, type=int)
    parser.add_argument("--lik_int_method", default="GH-20", type=str)
    parser.add_argument("--unroll", default=10, type=int)
    parser.add_argument("--joint_samples", dest="joint_samples", action="store_true")
    parser.set_defaults(joint_samples=False)
    
    parser.add_argument("--jitter", default=1e-6, type=float)
    parser.add_argument("--batch_size", default=10000, type=int)

    parser.add_argument("--lr_start", default=1e-2, type=float)
    parser.add_argument("--lr_end", default=1e-4, type=float)
    parser.add_argument("--lr_decay", default=0.98, type=float)
    parser.add_argument("--freeze_params", default=[], nargs="+", type=str)

    parser.add_argument("--max_epochs", default=3000, type=int)
    parser.add_argument("--loss_margin", default=-1e0, type=float)
    parser.add_argument("--margin_epochs", default=100, type=int)

    parser.add_argument("--likelihood", action="store", type=str)
    parser.add_argument("--filter_type", default="", action="store", type=str)
    parser.add_argument("--observations", action="store", type=str)
    parser.add_argument("--observed_covs", default="", action="store", type=str)
    parser.add_argument("--latent_covs", default="", action="store", type=str)
    return parser


### kernel ###
def covariate_kernel(kernel_dicts, f_dims, array_type):
    """
    Helper function for creating kernel triplet tuple
    """
    track_dims = 0
    dims_list, kernel_list = [], []
    for k, k_dict in enumerate(kernel_dicts):

        kernel_type = k_dict["type"]
        var_f = k_dict["var"]  # kernel variance (f_dims,)

        act = []
        for _ in range(k_dict["in_dims"]):
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
                krn = lib.GP.kernels.SquaredExponential(
                    f_dims,
                    var_f,
                    len_fx,
                    array_type=array_type,
                )

            elif kernel_type == "periodic":
                krn = lib.GP.kernels.Periodic(
                    f_dims, var_f, len_fx, array_type=array_type
                )

            elif kernel_type == "RQ":
                scale_mixture = k_dict["scale"]
                krn = kernels.RationalQuadratic(
                    f_dims, var_f, len_fx, scale_mixture, array_type=array_type
                )

            elif kernel_type == "matern12":
                krn = lib.GP.kernels.Matern12(
                    f_dims,
                    var_f,
                    len_fx,
                    array_type=array_type,
                )

            elif kernel_type == "matern32":
                krn = lib.GP.kernels.Matern32(
                    f_dims,
                    var_f,
                    len_fx,
                    array_type=array_type,
                )

            elif kernel_type == "matern52":
                krn = lib.GP.kernels.Matern52(
                    f_dims,
                    var_f,
                    len_fx,
                    array_type=array_type,
                )

            else:
                raise NotImplementedError("Kernel type is not supported.")

        kernel_list.append(krn)

    return kernel_list, dims_list


def ISI_kernel_dict_induc_list(rng, isi_order, num_induc, out_dims, kernel_types):
    """
    Kernel for time since spike and ISI dimensions

    Note all ISIs are transformed to [0, 1] range, so lengthscales based on that
    """
    induc_list = []
    kernel_dicts = []

    ones = np.ones(out_dims)

    for k in range(isi_order):
        order_arr = rng.permuted(
            np.tile(np.arange(num_induc), out_dims).reshape(out_dims, num_induc), 
            axis=1, 
        )
        len_t = (0.3 if k == 0 else 1.0) * np.ones((out_dims, 1))  # primary tuning to t since
        
        induc_list += [np.linspace(0.0, 1.0, num_induc)[order_arr][..., None]]
        kernel_dicts += [
            {"type": kernel_types[k], "in_dims": 1, "var": ones, "len": len_t}
        ]

    return kernel_dicts, induc_list


def latent_kernel_dict_induc_list(rng, latent_covs, num_induc, out_dims):
    """
    Construct kernel tuples for latent space
    """
    latent_covs_comps = latent_covs.split("-")[:-4]

    induc_list = []
    kernel_dicts = []

    if latent_covs != "":
        for zc in latent_covs_comps:
            d_z = int(zc.split("d")[-1])
            
            order_arr = rng.permuted(
                np.tile(np.arange(num_induc), out_dims*d_z).reshape(out_dims, d_z, num_induc), 
                axis=2, 
            )

            induc_list += [
                np.linspace(-2., 2., num_induc)[order_arr].transpose(0, 2, 1)
            ]
            ls = np.ones((out_dims, d_z))
            var = np.ones(d_z)
            kernel_dicts += [{"type": "SE", "in_dims": d_z, "var": var, "len": ls}]

    return kernel_dicts, induc_list


def compactify_kernel_dicts(kernel_dicts, concat_params):
    """
    Stack together parameters for consecutive kernels of same type
    """
    compact_kernel_dicts = [kernel_dicts[0]]

    for kd in kernel_dicts[1:]:
        if kd["type"] == compact_kernel_dicts[-1]["type"]:
            for name in concat_params:
                recent_dict = compact_kernel_dicts[-1]
                recent_dict[name] = np.concatenate(
                    [recent_dict[name], kd[name]], axis=-1
                )

            recent_dict["in_dims"] += kd["in_dims"]

        else:
            compact_kernel_dicts += [kd]

    return compact_kernel_dicts


def build_kernel(
    rng,
    observations,
    observed_covs,
    latent_covs,
    f_dims,
    isi_order,
    ISI_kernel_types, 
    gen_obs_kernel_induc_func,
    array_type,
):
    """
    Assemble the kernel components
    """
    observations_comps = observations.split("-")
    num_induc = int(observations_comps[1])

    kernel_dicts, induc_list = [], []

    if observations_comps[0] == "nonparam_pp_gp":  # ISIs
        kernel_dicts_isi, induc_list_isi = ISI_kernel_dict_induc_list(
            rng, isi_order, num_induc, f_dims, ISI_kernel_types
        )
        kernel_dicts += kernel_dicts_isi
        induc_list += induc_list_isi

    # observations
    obs_kernel_dicts, obs_induc_list = gen_obs_kernel_induc_func(
        rng, observed_covs, num_induc, f_dims
    )
    kernel_dicts += obs_kernel_dicts
    induc_list += obs_induc_list

    # latent
    kernel_dicts_lat, induc_list_lat = latent_kernel_dict_induc_list(
        rng, latent_covs, num_induc, f_dims
    )
    kernel_dicts += kernel_dicts_lat
    induc_list += induc_list_lat

    # product kernel
    kernel_dicts = compactify_kernel_dicts(kernel_dicts, ["len"])
    kernel_list, dims_list = covariate_kernel(kernel_dicts, f_dims, array_type)
    cov_kernel = lib.GP.kernels.Product(kernel_list, dims_list)
    induc_locs = np.concatenate(induc_list, axis=-1)  # (f_dims, num_induc, x_dims)
    return cov_kernel, induc_locs


### observation models ###
def build_spikefilters(rng, obs_dims, filter_type, array_type):
    """
    Create the spike coupling object
    """
    if filter_type == "":
        return None
    filter_type_comps = filter_type.split("-")
    filter_length = int(filter_type.split("H")[-1])

    if filter_type_comps[0] == "sigmoid":
        strs = filter_type_comps[1:4]
        alpha, beta, tau = int(strs[0]), float(strs[1]), float(strs[2])
        
        alpha = alpha * np.ones((obs_dims, 1))
        beta = beta * np.ones((obs_dims, 1))
        tau = tau * np.ones((obs_dims, 1))

        flt = lib.filters.SigmoidRefractory(
            alpha,
            beta,
            tau,
            filter_length,
            array_type, 
        )

    elif filter_type_comps[0] == "rcb":
        strs = filter_type_comps[1:6]
        B, phi_lower, phi_upper, a, c = (
            int(strs[0]), 
            float(strs[1].replace("n", "-")), 
            float(strs[2].replace("n", "-")), 
            float(strs[3]), 
            float(strs[4]), 
        )

        ini_var = 0.01
        if filter_type_comps[6] == "full":
            a = a * np.ones((obs_dims, obs_dims))
            c = c * np.ones((obs_dims, obs_dims))
            phi_h = np.broadcast_to(np.linspace(phi_lower, phi_upper, B), (B, obs_dims, obs_dims))
            w_h = np.sqrt(ini_var) * rng.normal(size=(B, obs_dims, obs_dims))

        elif filter_type_comps[6] == "self":
            a = a * np.ones((obs_dims, 1))
            c = c * np.ones((obs_dims, 1))
            phi_h = np.linspace(phi_lower, phi_upper, B)[:, None, None].repeat(obs_dims, axis=1)
            w_h = (np.sqrt(ini_var) * rng.normal(size=(B, obs_dims)))[..., None]

        else:
            raise ValueError('Invalid filter coupling mode (self or full)')

        flt = lib.filters.RaisedCosineBumps(
            a,
            c,
            w_h,
            phi_h,
            filter_length,
            array_type, 
        )

    elif filter_type_comps[0] == "svgp":
        strs = filter_type_comps[1:4]
        num_induc, a_r, tau_r = (
            int(strs[0]), float(strs[1].replace("n", "-")), float(strs[2])
        )

        if filter_type_comps[4] == "full":
            D = obs_dims ** 2
            a_r = a_r * np.ones((D, D))
            tau_r = tau_r * np.ones((D, D))

        elif filter_type_comps[4] == "self":
            D = obs_dims
            a_r = a_r * np.ones((D, 1))
            tau_r = tau_r * np.ones((D, 1))
            
        else:
            raise ValueError('Invalid filter coupling mode (self or full)')

        # qSVGP inducing points
        induc_locs = np.linspace(0, filter_length, num_induc)[None, :, None].repeat(D, axis=0)
        u_mu = 0.0 * rng.normal(size=(D, num_induc, 1))
        u_Lcov = 0.1 * np.eye(num_induc)[None, ...].repeat(D, axis=0)

        # kernel
        len_fx = filter_length / 4. * np.ones((D, 1))  # GP lengthscale
        beta = 0.0 * np.ones(D)
        len_beta = 1.5 * len_fx
        var_f = 0.1 * np.ones(D)  # kernel variance
        
        kern = lib.GP.kernels.DecayingSquaredExponential(
            D, variance=var_f, lengthscale=len_fx, 
            beta=beta, lengthscale_beta=len_beta, array_type=array_type)
        gp = lib.GP.sparse.qSVGP(
            kern, induc_locs, u_mu, u_Lcov, RFF_num_feats=0, whitened=True)

        flt = lib.filters.GaussianProcess(
            gp, 
            a_r,
            tau_r, 
            filter_length,
        )

    else:
        raise ValueError('Invalid filter coupling type')

    return flt


def build_factorized_gp(
    rng,
    gen_obs_kernel_induc_func,
    observed_covs,
    latent_covs,
    observations,
    likelihood,
    tbin,
    obs_dims,
    obs_filter,
    array_type,
):
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
    kernel, induc_locs = build_kernel(
        rng,
        observations,
        observed_covs,
        latent_covs,
        f_dims,
        0,
        None, 
        gen_obs_kernel_induc_func,
        array_type,
    )
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims

    # SVGP
    RFF_num_feats = int(observations.split("-")[2])
    num_induc = induc_locs.shape[1]
    u_mu = 0.01 * rng.normal(size=(f_dims, num_induc, 1))
    u_Lcov = 1.0 * np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)

    gp_mean = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(
        kernel, induc_locs, u_mu, u_Lcov, RFF_num_feats=RFF_num_feats, whitened=True
    )

    # model
    model = ModulatedFactorized(
        svgp, gp_mean, likelihood, spikefilter=obs_filter
    )
    return model


def build_renewal_gp(
    rng,
    gen_obs_kernel_induc_func,
    observed_covs,
    latent_covs,
    observations,
    likelihood,
    dt,
    obs_dims,
    obs_filter,
    model_type,
    mean_ISIs, 
    array_type,
):
    """
    Build models based on renewal densities
    """
    # likelihood
    likelihood_comps = likelihood.split("-")
    renewal_type = likelihood_comps[0]
    link_type = likelihood_comps[1]

    if renewal_type == "gamma":
        alpha = np.ones(obs_dims)
        renewal = lib.likelihoods.GammaRenewal(
            obs_dims,
            dt,
            alpha,
            link_type,
            array_type, 
        )

    elif renewal_type == "lognorm":
        sigma = np.ones(obs_dims)
        renewal = lib.likelihoods.LogNormalRenewal(
            obs_dims,
            dt,
            sigma,
            link_type,
            array_type, 
        )

    elif renewal_type == "invgauss":
        mu = np.ones(obs_dims)
        renewal = lib.likelihoods.InverseGaussianRenewal(
            obs_dims,
            dt,
            mu,
            link_type,
            array_type, 
        )
        
    elif renewal_type == "expon":
        renewal = lib.likelihoods.ExponentialRenewal(
            obs_dims,
            dt,
            link_type,
            array_type, 
        )

    else:
        raise ValueError("Invalid renewal likelihood")

    # kernel
    kernel, induc_locs = build_kernel(
        rng,
        observations,
        observed_covs,
        latent_covs,
        obs_dims,
        0,
        None, 
        gen_obs_kernel_induc_func,
        array_type,
    )
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims

    # SVGP
    RFF_num_feats = int(observations.split("-")[2])
    num_induc = induc_locs.shape[1]
    u_mu = 0.01 * rng.normal(size=(f_dims, num_induc, 1))
    u_Lcov = 1.0 * np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)

    gp_mean = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(
        kernel, induc_locs, u_mu, u_Lcov, RFF_num_feats=RFF_num_feats, whitened=True
    )

    # model
    if model_type == "rescaled":
        model = RateRescaledRenewal(
            svgp, gp_mean, renewal, spikefilter=obs_filter
        )

    elif model_type == "modulated":
        scale_tau = mean_ISIs
        
        model = ModulatedRenewal(
            svgp, gp_mean, scale_tau, renewal, spikefilter=obs_filter
        )

    else:
        raise ValueError("Invalid renewal model type")

    return model


def build_nonparametric(
    rng,
    gen_obs_kernel_induc_func,
    observed_covs,
    latent_covs,
    observations,
    likelihood,
    dt,
    obs_dims,
    mean_ISIs, 
    array_type,
):
    """
    Build point process models with nonparametric CIFs

    :param jnp.ndarray site_locs: temporal inducing point locations (f_dims, num_induc)
    :param float dt: time bin size for spike train data
    """
    # kernels
    observations_comps = observations.split("-")
    tau_kernel_type, ISI_kernel_type = observations_comps[2], observations_comps[3]
    RFF_num_feats = int(observations_comps[4])
    mean_amp = float(observations_comps[5].replace("n", "-"))

    isi_order = int(likelihood[3:])
    ISI_kernel_types = [tau_kernel_type] + [ISI_kernel_type] * (isi_order - 1)
    st_kernel, induc_locs = build_kernel(
        rng,
        observations,
        observed_covs,
        latent_covs,
        obs_dims,
        isi_order,
        ISI_kernel_types, 
        gen_obs_kernel_induc_func,
        array_type,
    )

    # SVGP
    num_induc = induc_locs.shape[1]
    u_mu = 0.01 * rng.normal(size=(obs_dims, num_induc, 1))
    u_Lcov = 1.0 * np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

    gp = lib.GP.sparse.qSVGP(
        st_kernel,
        induc_locs,
        u_mu,
        u_Lcov,
        RFF_num_feats=RFF_num_feats,
        whitened=True,
    )

    # BNPP
    wrap_tau = mean_ISIs * np.ones((obs_dims,))  # seconds
    mean_tau = mean_ISIs / 30. * np.ones((obs_dims,))
    mean_amp = mean_amp * np.ones((obs_dims,))
    mean_bias = 0.0 * np.ones((obs_dims,))

    model = NonparametricPointProcess(
        gp, wrap_tau, mean_tau, mean_amp, mean_bias, dt
    )
    return model


### model ###
def setup_latents(rng, d_x, latent_covs, site_lims, array_type):
    """
    latent covariates
    """
    latent_covs_comps = latent_covs.split("-")
    
    if len(latent_covs_comps) > 4:
        # settings
        num_site_locs = int(latent_covs_comps[-4])
        diagonal_site = latent_covs_comps[-3] == "diagonal_sites"
        diagonal_cov = latent_covs_comps[-2] == "diagonal_cov"
        fixed_grid_locs = latent_covs_comps[-1] == "fixed_grid"

        site_locs = np.linspace(site_lims[0], site_lims[-1], num_site_locs)

        # list
        tot_d_z, ss_kernels = 0, []
        for zc in latent_covs_comps[:-4]:
            d_z = 0
            ztype = zc.split("d")[0]

            if ztype == "matern12":
                d_z = int(zc[9:])

                var_z = 1.0 * np.ones((d_z))  # GP variance
                len_z = 1.0 * np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(
                    lib.GP.kernels.Matern12(
                        d_z, variance=var_z, lengthscale=len_z, array_type=array_type)
                )

            elif ztype == "matern32":
                d_z = int(zc[9:])

                var_z = 1.0 * np.ones((d_z))  # GP variance
                len_z = 1.0 * np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(
                    lib.GP.kernels.Matern32(
                        d_z, variance=var_z, lengthscale=len_z, array_type=array_type)
                )

            elif ztype == "matern52":
                d_z = int(zc[9:])

                var_z = 1.0 * np.ones((d_z))  # GP variance
                len_z = 1.0 * np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(
                    lib.GP.kernels.Matern52(
                        d_z, variance=var_z, lengthscale=len_z, array_type=array_type)
                )

            elif ztype == "LEG":
                d_z, d_s = [int(d) for d in latent_covs.split("s")]

                N = np.ones(d_s)[None]
                R = np.eye(d_s)[None]
                H = rng.normal(size=(d_z, d_s))[None] / np.sqrt(d_s)
                Lam = rng.normal(size=(d_s, d_s))[None] / np.sqrt(d_s)

                ss_kernels.append(lib.GP.kernels.LEG(N, R, H, Lam, array_type=array_type))

            elif zc != "":
                raise ValueError("Invalid latent covariate type")

            tot_d_z += d_z

            if len(ss_kernels) > 1:
                ss_kernels = lib.GP.kernels.StackMarkovian(ss_kernels)
            else:
                ss_kernels = ss_kernels[0]

            # site_init
            Tsteps = len(site_locs)
            site_obs = 0.0 * np.ones([Tsteps, tot_d_z, 1]) + 0 * rng.normal(
                size=(Tsteps, tot_d_z, 1)
            )
            site_Lcov = 1.0 * np.eye(tot_d_z)[None, ...].repeat(Tsteps, axis=0)

            ssgp = lib.GP.markovian.GaussianLTI(
                ss_kernels,
                site_locs,
                site_obs,
                site_Lcov,
                diagonal_site,
                fixed_grid_locs,
            )

    else:
        tot_d_z, ssgp, diagonal_cov = 0, None, True

    # joint latent observed covariates
    lat_covs_dims = list(range(d_x, d_x + tot_d_z))
    obs_covs_dims = list(range(d_x))
    
    inputs_model = GaussianLatentObservedSeries(
        ssgp, lat_covs_dims, obs_covs_dims, diagonal_cov, array_type
    )
    return inputs_model


def setup_observations(
    rng,
    gen_obs_kernel_induc_func,
    likelihood,
    filter_type,
    observations,
    observed_covs,
    latent_covs,
    obs_dims,
    tbin,
    mean_ISIs, 
    array_type,
):
    """
    Assemble the encoding model

    :return:
        used covariates, inputs model, observation model
    """
    ### GP observation model ###
    obs_filter = build_spikefilters(rng, obs_dims, filter_type, array_type)
    observations_comps = observations.split("-")

    if observations_comps[0] == "factorized_gp":
        obs_model = build_factorized_gp(
            rng,
            gen_obs_kernel_induc_func,
            observed_covs,
            latent_covs,
            observations,
            likelihood,
            tbin,
            obs_dims,
            obs_filter,
            array_type,
        )

    elif observations_comps[0] == "rate_renewal_gp":
        obs_model = build_renewal_gp(
            rng,
            gen_obs_kernel_induc_func,
            observed_covs,
            latent_covs,
            observations,
            likelihood,
            tbin,
            obs_dims,
            obs_filter,
            "rescaled",
            None, 
            array_type,
        )

    elif observations_comps[0] == "mod_renewal_gp":
        obs_model = build_renewal_gp(
            rng,
            gen_obs_kernel_induc_func,
            observed_covs,
            latent_covs,
            observations,
            likelihood,
            tbin,
            obs_dims,
            obs_filter,
            "modulated",
            mean_ISIs, 
            array_type,
        )

    elif observations_comps[0] == "nonparam_pp_gp":
        obs_model = build_nonparametric(
            rng,
            gen_obs_kernel_induc_func,
            observed_covs,
            latent_covs,
            observations,
            likelihood,
            tbin,
            obs_dims,
            mean_ISIs, 
            array_type,
        )

    else:
        raise ValueError("Invalid observation model type")

    return obs_model


### main functions ###
def select_inputs(dataset_dict, config):
    """
    Create the inputs to the model, all NumPy arrays

    Select the spike train to match section of covariates
    The spike history filter will add a buffer section at start
    Note the entire spike train is retained here from the dataset
    """
    observed_covs, observations, likelihood = (
        config.observed_covs,
        config.observations,
        config.likelihood,
    )

    ### ISIs ###
    if likelihood[:3] == "isi":
        ISI_order = int(likelihood[3:])
        ISIs = dataset_dict["ISIs"][..., :ISI_order]

    elif observations.split("-")[0] == "mod_renewal_gp":
        ISIs = dataset_dict["ISIs"][..., :1]

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
    ts = covs.shape[0]

    filter_length = (
        int(config.filter_type.split("H")[-1]) if config.filter_type != "" else 0
    )
    align_start_ind = dataset_dict["align_start_ind"]
    align_end_ind = align_start_ind + ts
    if align_start_ind - filter_length < 0:
        raise ValueError('Spike filter history exceeds available spike train')
    ttslice = slice(align_start_ind - filter_length, align_end_ind)

    ys = dataset_dict["spiketrains"][:, ttslice].astype(float)
    timestamps = dataset_dict["timestamps"]
    ISIs = (
        ISIs.transpose(1, 0, 2) if ISIs is not None else None
    )  # (obs_dims, ts, order)

    return timestamps, covs, ISIs, ys, filter_length


def build_model(
    config,
    dataset_dict,
    observed_kernel_dict_induc_list,
    rng,
    timestamps,
    obs_covs_dims,
):
    if config.double_arrays:
        array_type = "float64"
    else:
        array_type = "float32"

    gen_kernel_induc_func = (
        lambda rng, observations, num_induc, out_dims: observed_kernel_dict_induc_list(
            rng, observations, num_induc, out_dims, dataset_dict["covariates"]
        )
    )

    obs_dims = dataset_dict["properties"]["neurons"]
    tbin = float(dataset_dict["properties"]["tbin"])
    
    ISIs = dataset_dict["ISIs"]
    mean_ISIs = np.array([np.unique(ISIs[:, n, 1]).mean() for n in range(ISIs.shape[1])])

    # create and initialize model
    inp_model = setup_latents(
        rng,
        obs_covs_dims,
        config.latent_covs,
        [timestamps[0], timestamps[-1]],
        array_type,
    )

    obs_model = setup_observations(
        rng,
        gen_kernel_induc_func,
        config.likelihood,
        config.filter_type,
        config.observations,
        config.observed_covs,
        config.latent_covs,
        obs_dims,
        tbin,
        mean_ISIs, 
        array_type,
    )
    model = GaussianTwoLayer(inp_model, obs_model)

    return model


def setup_jax(config):
    if config.force_cpu:
        jax.config.update("jax_platform_name", "cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.device}"

    if config.double_arrays:
        jax.config.update("jax_enable_x64", True)
      
    
    

def gen_name(parser_args, dataset_dict):

    name = dataset_dict["properties"]["name"] + "_{}_{}_{}_X[{}]_Z[{}]_freeze[{}]".format(
        parser_args.likelihood,
        parser_args.filter_type,
        parser_args.observations + ('jointsamples' if parser_args.joint_samples else ''),
        parser_args.observed_covs,
        parser_args.latent_covs,
        "-".join(parser_args.freeze_params).replace(".", "0"), 
    )
    return name

    

def fit_and_save(parser_args, dataset_dict, observed_kernel_dict_induc_list, save_name):
    """
    Fit and save the model
    General training loop
    """
    config = argparse.Namespace(**{**vars(parser_args), **dataset_dict["properties"]})

    # JAX
    setup_jax(config)

    # data preparation
    timestamps, covariates, ISIs, observations, filter_length = select_inputs(
        dataset_dict, config
    )
    obs_covs_dims = covariates.shape[-1]
    
    tot_ts = len(timestamps)
    dataloader = BatchedTimeSeries(
        timestamps, covariates, ISIs, observations, config.batch_size, filter_length
    )

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

        rng = np.random.default_rng(seed)
        model = build_model(
            config,
            dataset_dict,
            observed_kernel_dict_induc_list,
            rng,
            timestamps,
            obs_covs_dims,
        )

        # freeze parameters
        select_fixed_params = lambda tree: [
            rgetattr(tree, name) for name in config.freeze_params
        ]

        filter_spec = jax.tree_map(lambda o: eqx.is_inexact_array(o), model)
        filter_spec = eqx.tree_at(
            select_fixed_params,
            filter_spec,
            replace=(False,) * len(config.freeze_params),
        )

        # loss
        @eqx.filter_value_and_grad
        def compute_loss(diff_model, static_model, prng_state, num_samps, jitter, data, lik_int_method):
            model = eqx.combine(diff_model, static_model)
            nELBO = -model.ELBO(
                prng_state, num_samps, jitter, tot_ts, data, lik_int_method, 
                config.joint_samples, config.unroll
            )
            return nELBO

        @partial(eqx.filter_jit, device=jax.devices()[0])
        def make_step(
            model, prng_state, num_samps, jitter, data, lik_int_method, opt_state
        ):
            diff_model, static_model = eqx.partition(model, filter_spec)
            loss, grads = compute_loss(
                diff_model, static_model, prng_state, num_samps, jitter, data, lik_int_method
            )

            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            model = model.apply_constraints()
            return loss, model, opt_state

        lik_int_comps = config.lik_int_method.split("-")
        lik_int_method = {
            "type": lik_int_comps[0],
            "approx_pts": int(lik_int_comps[1]),
        }

        # initialize optimizers
        opt_state = optim.init(model)
        loss_tracker = {
            "train_loss_batches": [],
            "train_loss_epochs": [],
        }
        lrs = []

        # try:  # attempt to fit model
        prng_state = jr.PRNGKey(seed)

        minloss = np.inf
        iterator = tqdm(range(config.max_epochs))
        for epoch in iterator:

            avg_loss = []
            for b in range(dataloader.batches):
                batch_data = dataloader.load_batch(b)
                prng_state, prng_key = jr.split(prng_state)

                loss, model, opt_state = make_step(
                    model,
                    prng_key,
                    config.num_MC,
                    config.jitter,
                    batch_data,
                    lik_int_method,
                    opt_state,
                )
                loss = loss.item()
                avg_loss.append(loss)

                loss_dict = {"train_loss_batches": loss}
                for n, v in loss_dict.items():
                    loss_tracker[n].append(v)
                lrs.append(learning_rate_schedule(epoch).item())

                iterator.set_postfix(**loss_dict)

            avgbloss = (
                np.array(avg_loss).mean().item()
            )  # average over batches (subsampled estimator of loss)
            loss_dict = {"train_loss_epochs": avgbloss}
            for n, v in loss_dict.items():
                loss_tracker[n].append(v)

            if avgbloss <= minloss + config.loss_margin:
                cnt = 0
            else:
                cnt += 1

            if avgbloss < minloss:
                minloss = avgbloss

            if cnt > config.margin_epochs:
                print("Stopped at epoch {}.".format(epoch + 1))
                break

        # save and progress
        savefile = config.checkpoint_dir + save_name

        if os.path.exists(savefile + ".p"):  # check previous best losses
            with open(savefile + ".p", "rb") as f:
                results = pickle.load(f)
                final_loss = results["losses"]["train_loss_epochs"][-1]
        else:
            final_loss = np.inf  # nonconvex optimization, pick the best

        if avgbloss < final_loss:
            if not os.path.exists(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)

            # save model
            eqx.tree_serialise_leaves(savefile + ".eqx", model)

            # save results
            savedata = {
                "losses": loss_tracker,
                "lrs": lrs,
                "best_seed": seed,
                "config": config,
            }
            with open(savefile + ".p", "wb") as f:
                pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)

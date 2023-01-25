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

sys.path.append("..")
import lib

from lib.inference.base import Observations
from lib.inference.svgp import (
    ModulatedFactorized,
    ModulatedRenewal,
    NonparametricPointProcess,
    RateRescaledRenewal,
)
from lib.inference.timeseries import GaussianLatentObservedSeries


### script ###
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


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
    parser.add_argument("--array_type", default="float32", type=str)

    parser.add_argument("--seeds", default=[123], nargs="+", type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--num_MC", default=1, type=int)
    parser.add_argument("--lik_int_method", default="GH-20", type=str)

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
                    metric_type="Euclidean",
                    array_type=array_type,
                )

            elif kernel_type == "circSE":
                krn = lib.GP.kernels.SquaredExponential(
                    f_dims, var_f, len_fx, metric_type="Cosine", array_type=array_type
                )

            elif kernel_type == "RQ":
                scale_mixture = k_dict["scale"]
                krn = kernels.RationalQuadratic(
                    f_dims, var_f, len_fx, scale_mixture, array_type=array_type
                )

            elif kernel_type == "Matern12":
                krn = lib.GP.kernels.Matern12(
                    f_dims,
                    var_f,
                    len_fx,
                    array_type=array_type,
                )

            elif kernel_type == "Matern32":
                krn = lib.GP.kernels.Matern32(
                    f_dims,
                    var_f,
                    len_fx,
                    array_type=array_type,
                )

            elif kernel_type == "Matern52":
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


def ISI_kernel_dict_induc_list(rng, isi_order, num_induc, out_dims):
    """
    Kernel for time since spike and ISI dimensions

    Note all ISIs are transformed to [0, 1] range, so lengthscales based on that
    """
    induc_list = []
    kernel_dicts = []

    ones = np.ones(out_dims)

    for _ in range(isi_order - 1):  # first time to spike separate
        induc_list += [rng.normal(size=(out_dims, num_induc, 1))]
        kernel_dicts += [
            {"type": "SE", "in_dims": 1, "var": ones, "len": np.ones((out_dims, 1))}
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
            induc_list += [rng.normal(size=(out_dims, num_induc, d_z))]
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
            rng, isi_order, num_induc, f_dims
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
def build_spikefilters(rng, obs_dims, filter_type):
    """
    Create the spike coupling object.

    if spkcoupling_mode[:2] == 'gp':
        l0, l_b0, beta0, l_mean0, num_induc = filter_props
        if spkcoupling_mode[2:6] == 'full':
            D = obs_dims*obs_dims
        else:
            D = obs_dims
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
    filter_type_comps = filter_type.split("-")
    filter_length = int(filter_type.split("H")[-1])

    if filter_type_comps[0] == "sigmoid":
        alpha = np.ones((obs_dims, 1))
        beta = np.ones((obs_dims, 1))
        tau = 10 * np.ones((obs_dims, 1))

        flt = lib.filters.FIR.SigmoidRefractory(
            alpha,
            beta,
            tau,
            filter_length,
        )

    elif filter_type_comps[0] == "rcb":
        strs = filter_type_comps[1:5]
        B, L, a, c = int(strs[0]), float(strs[1]), float(strs[2]), float(strs[3])

        ini_var = 1.0
        if filter_type_comps[5] == "full":
            a = a * np.ones((2, obs_dims, obs_dims))
            c = c * np.ones((2, obs_dims, obs_dims))
            phi_h = np.broadcast_to(np.linspace(0.0, L, B), (B, obs_dims, obs_dims))
            w_h = np.sqrt(ini_var) * rng.normal(size=(B, obs_dims, obs_dims))

        elif filter_type_comps[5] == "self":
            a = a * np.ones((2, obs_dims, 1))
            c = c * np.ones((2, obs_dims, 1))
            phi_h = np.linspace(0.0, L, B)[:, None, None].repeat(obs_dims, axis=1)
            w_h = (np.sqrt(ini_var) * rng.normal(size=(B, obs_dims)))[..., None]

        else:
            raise ValueError

        flt = lib.filters.FIR.RaisedCosineBumps(
            a,
            c,
            w_h,
            phi_h,
            filter_length,
        )

    elif filter_type_comps[0] == "svgp":
        num_induc = int(filter_type_comps[1])

        if filter_type_comps[2] == "full":
            D = obs_dims * obs_dims
            out_dims = obs_dims

        elif filter_type_comps[2] == "self":
            D = obs_dims
            out_dims = 1

        else:
            raise ValueError

        v = 100 * tbin * torch.ones(1, D)
        l = 100.0 * tbin * torch.ones((1, D))
        l_b = 100.0 * tbin * torch.ones((1, D))
        beta = 1.0 * torch.ones((1, D))

        mean_func = nppl.kernels.means.decaying_exponential(D, 0.0, 100.0 * tbin)

        Xu = torch.linspace(0, hist_len * tbin, num_induc)[None, :, None].repeat(
            D, 1, 1
        )

        krn2 = lib.GP.kernels.DecayingSquaredExponential(
            input_dims=1,
            lengthscale=l,
            lengthscale_beta=l_b,
            beta=beta,
            track_dims=[0],
            f="exp",
            tensor_type=tensor_type,
        )
        kernelobj = lib.GP.kernel.Product(krn1, krn2)
        inducing_points = jnp.array(Xu)

        gp = lib.GP.sparse.qSVGP(
            1,
            D,
            kernelobj,
            inducing_points=inducing_points,
            whitened=True,
        )

        flt = lib.filters.GaussianProcess(out_dims, obs_dims, hist_len + 1, tbin, gp)

    else:
        raise ValueError

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
        gen_obs_kernel_induc_func,
        array_type,
    )
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims

    # SVGP
    RFF_num_feats = int(observations.split("-")[2])
    num_induc = induc_locs.shape[1]
    u_mu = 1.0 + 0.0 * rng.normal(size=(f_dims, num_induc, 1))
    u_Lcov = 0.01 * np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)

    gp_mean = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(
        kernel, induc_locs, u_mu, u_Lcov, RFF_num_feats=RFF_num_feats, whitened=True
    )

    # model
    model = lib.inference.svgp.ModulatedFactorized(
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
        renewal = lib.likelihoods.Gamma(
            obs_dims,
            dt,
            alpha,
            link_type,
        )

    elif renewal_type == "lognorm":
        sigma = np.ones(obs_dims)
        renewal = lib.likelihoods.LogNormal(
            obs_dims,
            dt,
            sigma,
            link_type,
        )

    elif renewal_type == "invgauss":
        mu = np.ones(obs_dims)
        renewal = lib.likelihoods.InverseGaussian(
            obs_dims,
            dt,
            mu,
            link_type,
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
        gen_obs_kernel_induc_func,
        array_type,
    )
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims

    # SVGP
    RFF_num_feats = int(observations.split("-")[2])
    num_induc = induc_locs.shape[1]
    u_mu = 1.0 + 0.0 * rng.normal(size=(f_dims, num_induc, 1))
    u_Lcov = 0.01 * np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)

    gp_mean = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(
        kernel, induc_locs, u_mu, u_Lcov, RFF_num_feats=RFF_num_feats, whitened=True
    )

    # model
    if model_type == "rescaled":
        model = lib.inference.svgp.RateRescaledRenewal(
            svgp, gp_mean, renewal, spikefilter=obs_filter
        )

    elif model_type == "modulated":
        model = lib.inference.svgp.ModulatedRenewal(
            svgp, gp_mean, renewal, spikefilter=obs_filter
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
    array_type,
    stvgp=False,
):
    """
    Build point process models with nonparametric CIFs

    :param jnp.ndarray site_locs: temporal inducing point locations (f_dims, num_induc)
    :param float dt: time bin size for spike train data
    """
    # kernels
    observations_comps = observations.split("-")
    ss_type = observations_comps[2]
    RFF_num_feats = int(observations_comps[3])
    refract_neg = -float(observations_comps[4])  # -12.

    if ss_type == "matern12":
        len_t = 1.0 * np.ones((obs_dims, 1))  # GP lengthscale
        var_t = 1.0 * np.ones(obs_dims)  # GP variance
        ss_kernel = lib.GP.kernels.Matern12(obs_dims, variance=var_t, lengthscale=len_t)

    elif ss_type == "matern32":
        len_t = 1.0 * np.ones((obs_dims, 1))  # GP lengthscale
        var_t = 1.0 * np.ones(obs_dims)  # GP variance
        ss_kernel = lib.GP.kernels.Matern32(obs_dims, variance=var_t, lengthscale=len_t)

    elif ss_type == "matern52":
        len_t = 1.0 * np.ones((obs_dims, 1))  # GP lengthscale
        var_t = 1.0 * np.ones(obs_dims)  # GP variance
        ss_kernel = lib.GP.kernels.Matern52(obs_dims, variance=var_t, lengthscale=len_t)

    else:
        raise ValueError

    isi_order = int(likelihood[3:])
    kernel, induc_locs = build_kernel(
        rng,
        observations,
        observed_covs,
        latent_covs,
        obs_dims,
        isi_order,
        gen_obs_kernel_induc_func,
        array_type,
    )

    if stvgp:
        num_induc_t = int(observations_comps[5])
        spatial_MF = observations_comps[6] == "spatial_MF"
        fixed_grid_locs = observations_comps[7] == "fixed_grid"

        # spatiotemporal SVGP
        num_induc_sp = induc_locs.shape[1]
        num_induc = num_induc_sp * num_induc_t

        site_locs = np.linspace(0.0, 1.0, num_induc_t)[None, :].repeat(obs_dims, axis=0)
        site_obs = 0.1 * rng.normal(size=(obs_dims, num_induc_t, num_induc_sp, 1))
        if spatial_MF:
            site_Lcov = 0.1 * np.ones((1, 1, num_induc_sp, 1)).repeat(
                num_induc_t, axis=1
            ).repeat(obs_dims, axis=0)
        else:  # full posterior covariance
            site_Lcov = 0.1 * np.eye(num_induc_sp)[None, None, ...].repeat(
                num_induc_t, axis=1
            ).repeat(obs_dims, axis=0)

        st_kernel = lib.GP.kernels.MarkovSparseKronecker(ss_kernel, kernel, induc_locs)

        gp = lib.GP.spatiotemporal.KroneckerLTI(
            st_kernel,
            site_locs,
            site_obs,
            site_Lcov,
            RFF_num_feats,
            spatial_MF,
            fixed_grid_locs,
        )

    else:  # SVGP
        num_induc = induc_locs.shape[1]

        site_locs = np.linspace(0.0, 1.0, num_induc)[None, :, None].repeat(
            obs_dims, axis=0
        )
        induc_locs = jnp.concatenate((site_locs, induc_locs), axis=-1)
        st_kernel = lib.GP.kernels.Product(
            [ss_kernel, kernel], [[0], list(range(1, induc_locs.shape[-1]))]
        )

        u_mu = 1.0 * rng.normal(size=(obs_dims, num_induc, 1))
        u_Lcov = 0.1 * np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

        gp = lib.GP.sparse.qSVGP(
            st_kernel,
            induc_locs,
            u_mu,
            u_Lcov,
            RFF_num_feats=RFF_num_feats,
            whitened=True,
        )

    # BNPP
    wrap_tau = 1.0 * np.ones((obs_dims,))  # seconds
    refract_tau = 1e-1 * np.ones((obs_dims,))
    mean_bias = 0.0 * np.ones((obs_dims,))

    model = lib.inference.svgp.NonparametricPointProcess(
        gp, wrap_tau, refract_tau, refract_neg, mean_bias, dt
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
                    lib.GP.kernels.Matern12(d_z, variance=var_z, lengthscale=len_z)
                )

            elif ztype == "matern32":
                d_z = int(zc[9:])

                var_z = 1.0 * np.ones((d_z))  # GP variance
                len_z = 1.0 * np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(
                    lib.GP.kernels.Matern32(d_z, variance=var_z, lengthscale=len_z)
                )

            elif ztype == "matern52":
                d_z = int(zc[9:])

                var_z = 1.0 * np.ones((d_z))  # GP variance
                len_z = 1.0 * np.ones((d_z, 1))  # GP lengthscale
                ss_kernels.append(
                    lib.GP.kernels.Matern52(d_z, variance=var_z, lengthscale=len_z)
                )

            elif ztype == "LEG":
                d_z, d_s = [int(d) for d in latent_covs.split("s")]

                N = np.ones(d_s)[None]
                R = np.eye(d_s)[None]
                H = rng.normal(size=(d_z, d_s))[None] / np.sqrt(d_s)
                Lam = rng.normal(size=(d_s, d_s))[None] / np.sqrt(d_s)

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

    inputs_model = lib.inference.timeseries.GaussianLatentObservedSeries(
        ssgp, lat_covs_dims, obs_covs_dims, diagonal_cov
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
    array_type,
):
    """
    Assemble the encoding model

    :return:
        used covariates, inputs model, observation model
    """
    ### GP observation model ###
    obs_filter = build_spikefilters(rng, obs_dims, filter_type)
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
            array_type,
        )

    else:
        raise ValueError("Invalid observation model type")

    return obs_model


### main functions ###
def select_inputs(dataset_dict, config):
    """
    Create the inputs to the model

    Trim the spike train to match section of covariates
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

    # trim
    filter_length = (
        int(config.filter_type.split("H")[-1]) if config.filter_type != "" else 0
    )
    align_start_ind = dataset_dict["align_start_ind"]
    align_end_ind = align_start_ind + ts
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
    seed,
    timestamps,
    obs_covs_dims,
):
    gen_kernel_induc_func = (
        lambda rng, observations, num_induc, out_dims: observed_kernel_dict_induc_list(
            rng, observations, num_induc, out_dims, dataset_dict["covariates"]
        )
    )

    obs_dims = dataset_dict["properties"]["neurons"]
    tbin = float(dataset_dict["properties"]["tbin"])

    # seed numpy rng
    rng = np.random.default_rng(seed)

    # create and initialize model
    inp_model = setup_latents(
        rng,
        obs_covs_dims,
        config.latent_covs,
        [timestamps[0], timestamps[-1]],
        config.array_type,
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
        config.array_type,
    )
    model = lib.inference.svgp.GPLVM(inp_model, obs_model)

    return model


def fit_and_save(parser_args, dataset_dict, observed_kernel_dict_induc_list, save_name):
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
        jax.config.update("jax_enable_x64", True)

    # data preparation
    timestamps, covariates, ISIs, observations, filter_length = select_inputs(
        dataset_dict, config
    )
    obs_covs_dims = covariates.shape[-1]

    tot_ts = len(timestamps)
    dataloader = lib.inference.timeseries.BatchedTimeSeries(
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

        model = build_model(
            config,
            dataset_dict,
            observed_kernel_dict_induc_list,
            seed,
            timestamps,
            obs_covs_dims,
        )

        # freeze parameters
        select_fixed_params = lambda tree: [
            rgetattr(tree, name) for name in config.fix_param_names
        ]

        filter_spec = jax.tree_map(lambda o: eqx.is_inexact_array(o), model)
        filter_spec = eqx.tree_at(
            select_fixed_params,
            filter_spec,
            replace=(False,) * len(config.fix_param_names),
        )

        # loss
        @partial(eqx.filter_value_and_grad, arg=filter_spec)
        def compute_loss(model, prng_state, num_samps, jitter, data, lik_int_method):
            nELBO = -model.ELBO(
                prng_state, num_samps, jitter, tot_ts, data, lik_int_method
            )
            return nELBO

        @partial(eqx.filter_jit, device=jax.devices()[0])
        def make_step(
            model, prng_state, num_samps, jitter, data, lik_int_method, opt_state
        ):
            loss, grads = compute_loss(
                model, prng_state, num_samps, jitter, data, lik_int_method
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
                lrs.append(learning_rate_schedule(epoch))

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

        # except (ValueError, RuntimeError) as e:
        #    print(e)

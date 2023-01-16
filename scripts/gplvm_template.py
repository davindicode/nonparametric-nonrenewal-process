import argparse
import os

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



### script ###
def standard_parser(usage, description):
    """
    Parser arguments belonging to training loop
    """
    parser = argparse.ArgumentParser(usage=usage, description=description)
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

    parser.add_argument("--jitter", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=10000, type=int)

    parser.add_argument("--lr_start", default=1e-2, type=float)
    parser.add_argument("--lr_end", default=1e-4, type=float)
    parser.add_argument("--lr_decay", default=0.98, type=float)
    
    parser.add_argument("--max_epochs", default=3000, type=int)
    parser.add_argument("--loss_margin", default=-1e0, type=float)
    parser.add_argument("--margin_epochs", default=100, type=int)

    parser.add_argument("--likelihood", action="store", type=str)
    parser.add_argument("--filter", default="", action="store", type=str)
    parser.add_argument("--filter_length", default=0, type=int)
    parser.add_argument("--mapping", default="", action="store", type=str)
    parser.add_argument("--observed_cov", default="", action="store", type=str)
    parser.add_argument("--latent_cov", default="", action="store", type=str)
    return parser



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



def ISI_kernel_dict_induc_list(isi_order):
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


def latent_kernel_dict_induc_list(z_mode, num_induc, out_dims):
    """
    Construct kernel tuples for latent space
    """
    z_mode_comps = z_mode.split("-")

    induc_list = []
    kernel_dicts = []

    for zc in z_mode_comps:
        d_z = int(zc.split("d")[-1])
        induc_list += [np.random.randn(num_induc)] * d_z
        ls = np.ones((out_dims, d_z))
        var = np.ones(d_z)
        kernel_dicts += [{"type": "SE", "var": var, "len": ls}]

    return kernel_dicts, induc_list






### observation models ###
def build_spikefilters(filt_mode, filter_length):                   
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
    if filt_mode == 'sigmoid':
        alpha = np.ones((neurons, 1))
        beta = np.ones((neurons, 1))
        tau = 10*np.ones((neurons, 1))

        flt = lib.filters.FIR.SigmoidRefractory(
            alpha,
            beta,
            tau,
            filter_length,
        )
        
    elif filt_mode[4:8] == 'svgp':
        num_induc = int(filt_mode[8:])
        
        if filt_mode[:4] == 'full':
            D = neurons*neurons
            out_dims = neurons
        elif filt_mode[:4] == 'self':
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
        
        
    elif filt_mode[4:7] == 'rcb':
        strs = filt_mode[7:].split('-')
        B, L, a, c = int(strs[0]), float(strs[1]), torch.tensor(float(strs[2])), torch.tensor(float(strs[3]))
        
        ini_var = 1.
        if filt_mode[:4] == 'full':
            phi_h = np.linspace(0., L, B)[:, None, None].repeat(1, neurons, neurons)
            w_h = np.sqrt(ini_var) * np.random.randn(B, neurons, neurons)

        elif filt_mode[:4] == 'self':
            phi_h = np.linspace(0., L, B)[:, None].repeat(1, neurons)
            w_h = np.sqrt(ini_var) * np.random..randn(B, neurons)
            
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



def build_factorized_gp(cov_kernel, induc_list, ll_mode, tbin, obs_dims, obs_filter, jitter, array_type):
    """
    Build models with likelihoods that factorize over time steps
    """
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims
    
    ### SVGP ###
    num_induc = induc_locs.shape[1]
    u_mu = 1. + 0.*np.random.randn(prng_state, shape=(f_dims, num_induc, 1))
    u_Lcov = 0.01*np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)
    
    mean_f = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(kern, mean_f, induc_locs, u_mu, u_Lcov, 
                               RFF_num_feats=0, array_type=array_type)

    Kzz = svgp.kernel.K(svgp.induc_locs, None, False)
    lambda_1, chol_Lambda_2 = lib.GP.sparse.t_from_q_svgp_moments(Kzz, u_mu, u_Lcov)

    ll_mode_comps = ll_mode.split("-")
    lik_type = ll_mode_comps[0]

    # likelihood
    if lik_type == "IBP":
        likelihood = lib.likelihoods.factorized.Bernoulli(
            obs_dims, tbin, ll_mode_comps[-1], array_type=array_type
        )

    elif lik_type == "PP":
        likelihood = lib.likelihoods.PointProcess(
            obs_dims, tbin, ll_mode_comps[-1], array_type=array_type
        )
        
    elif lik_type == "IP":
        likelihood = lib.likelihoods.Poisson(
            obs_dims, tbin, ll_mode_comps[-1], array_type=array_type
        )

    elif lik_type == "ZIP":
        alpha = 0.1 * torch.ones(inner_dims)
        likelihood = lib.likelihoods.ZeroInflatedPoisson(
            obs_dims, tbin, alpha, ll_mode_comps[-1], array_type=array_type
        )

    elif lik_type == "hZIP":
        likelihood = lib.likelihoods.HeteroscedasticZeroInflatedPoisson(
            obs_dims, tbin, ll_mode_comps[-1], array_type=array_type
        )

    elif lik_type == "NB":
        r_inv = 10.0 * torch.ones(inner_dims)
        likelihood = lib.likelihoods.factorized.NegativeBinomial(
            obs_dims, tbin, r_inv, ll_mode_comps[-1], array_type=array_type
        )

    elif lik_type == "hNB":
        likelihood = lib.likelihoods.factorized.HeteroscedasticNegativeBinomial(
            obs_dims, tbin, ll_mode_comps[-1], array_type=array_type
        )

    elif lik_type == "CMP":
        J = int(ll_mode_comps[1])
        log_nu = torch.zeros(inner_dims)
        likelihood = lib.likelihoods.factorized.ConwayMaxwellPoisson(
            obs_dims, tbin, log_nu, ll_mode_comps[-1], J=J, array_type=array_type
        )

    elif lik_type == "hCMP":
        J = int(ll_mode_comps[1])
        likelihood = lib.likelihoods.factorized.HeteroscedasticConwayMaxwellPoisson(
            obs_dims, tbin, ll_mode_comps[-1], J=J, array_type=array_type
        )

    elif lik_type == "U":
        basis_mode = ll_mode_comps[1]
        C = int(ll_mode_comps[2])
        K = int(ll_mode_comps[3])
        
        likelihood = lib.likelihoods.factorized.UniversalCount(
            obs_dims, C, K, basis_mode, array_type=array_type
        )

    else:
        raise NotImplementedError
    
    model = lib.inference.gp.ModulatedFactorized(svgp, likelihood, spikefilter=obs_filter)
    return model



def build_renewal_gp(renewal_type, model_type, jitter, array_type):
    """
    Build models based on renewal densities
    """
    x_dims = kernel.in_dims
    f_dims = kernel.out_dims
    
    ### SVGP ###
    num_induc = induc_locs.shape[1]
    u_mu = 1. + 0.*np.random.randn(prng_state, shape=(f_dims, num_induc, 1))
    u_Lcov = 0.01*np.eye(num_induc)[None, ...].repeat(f_dims, axis=0)
    
    mean_f = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(kern, mean_f, induc_locs, u_mu, u_Lcov, 
                               RFF_num_feats=0, array_type=array_type)

    Kzz = svgp.kernel.K(svgp.induc_locs, None, False)
    lambda_1, chol_Lambda_2 = lib.GP.sparse.t_from_q_svgp_moments(Kzz, u_mu, u_Lcov)
    
    ### likelihood ###
    if renewal_type == 'gamma':
        alpha = np.linspace(0.5, 1.5, neurons)
        renewal = lib.likelihoods.Gamma(
            neurons,
            dt,
            alpha,
        )

    elif renewal_type == 'lognorm':
        sigma = np.linspace(0.5, 1.5, neurons)
        renewal = lib.likelihoods.LogNormal(
            neurons,
            dt,
            sigma,
        )

    elif renewal_type == 'invgauss':
        mu = np.linspace(0.5, 1.5, neurons)
        renewal = lib.likelihoods.InverseGaussian(
            neurons,
            dt,
            mu,
        )
        
    else:
        raise ValueError('Invalid renewal likelihood')

    if model_type == 'rescaled':
        model = lib.inference.gp.RateRescaledRenewal(svgp, renewal, spikefilter=flt)
        
    elif model_type == 'modulated':
        model = lib.inference.gp.ModulatedRenewal(svgp, renewal, spikefilter=flt)
        
    else:
        raise ValueError('Invalid renewal model type')
        
    return model



def build_nonparametric(ss_kernel, cov_kernel, induc_locs, site_locs, spatial_MF, fixed_grid_locs, jitter, array_type):
    """
    Build point process models with nonparametric CIFs
    """
    x_dims = ss_kernel.out_dims
    dt = 1e-3  # ms
    
#     if len(kernel_list) == 0:  # not modulated
#         Tsteps = 1000

#         site_locs = np.linspace(0., 1., Tsteps)[None, :].repeat(x_dims, axis=0)  # s
#         site_obs = 0. * np.ones([x_dims, Tsteps, 1]) + 0*np.random.randn(x_dims, Tsteps, 1)
#         site_Lcov = 1. * np.ones([x_dims, Tsteps, 1]) + 0*np.random.randn(x_dims, Tsteps, 1)

#         state_space = lib.GP.markovian.MultiOutputLTI(
#             ss_kernel, site_locs, site_obs, site_Lcov, fixed_grid_locs=True)
    
#     else:

    x_dims = cov_kernel.in_dims
    f_dims = cov_kernel.out_dims

    ### SVGP ###
    mean_f = np.zeros(f_dims)
    svgp = lib.GP.sparse.qSVGP(cov_kernel, mean_f, induc_locs, u_mu, u_Lcov, 
                               RFF_num_feats=0, array_type=array_type)

    ### SSGP ###
    num_induc_t = site_locs.shape[-1]
    num_induc_sp = induc_locs.shape[1]
    num_induc = num_induc_sp * num_induc_t

    #site_locs = np.linspace(0., 1., num_t_ind)[None, :].repeat(f_dims, axis=0)  # s

    site_obs = 0.1 * np.random.randn(f_dims, num_induc_t, num_induc_sp, 1)
    site_Lcov = 0.1 * np.eye(num_induc_sp)[None, None, ...].repeat(
        num_induc_t, axis=1).repeat(f_dims, axis=0)

    ### spatiotemporal GP ###
    st_kernel = lib.GP.kernels.MarkovSparseKronecker(ss_kernel, cov_kernel, induc_locs)
    spatiotemporal = lib.GP.spatiotemporal.KroneckerLTI(
        st_kernel, site_locs, site_obs, site_Lcov, spatial_MF, fixed_grid_locs)
    
    ### bnpp ###
    wrap_tau = 10.*np.ones((x_dims,))
    refract_tau = 1e0*np.ones((x_dims,))
    refract_neg= -12.
    mean_bias = 0.*np.ones((x_dims,))
    
    model = lib.inference.gp.NonparametricPointProcess(
        state_space, wrap_tau, refract_tau, refract_neg, mean_bias, dt)
    return model


### model ###
def setup_inputs(dataset_dict, x_mode, ll_mode, site_locs, diagonal_site, fixed_grid_locs, array_type):
    """
    Create the inputs to the model
    """
    
    ### ISIs ###
    if ll_mode[:3] == 'isi':
        ISI_order = int(ll_mode[3:])
        ISIs = dataset_dict["ISIs"][..., :ISI_order]
        
    else:
        ISIs = None
    
    ### observed covariates ###
    x_mode_comps = x_mode.split("-")
    covariates = dataset_dict["covariates"]
    
    input_data = []
    for xc in x_mode_comps:
        if xc == "":
            continue
        input_data.append(covariates[xc])

    return ISIs, input_data



def setup_latents(obs_dims, z_mode):
    ### latent covariates ###
    z_mode_comps = z_mode.split("-")

    tot_d_z, ss_kernels = 0, []
    for zc in z_mode_comps:
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
            d_z, d_s = [int(d) for d in z_mode.split("s")]
            
            N = np.ones(d_s)[None]
            R = np.eye(d_s)[None]
            H = np.random.randn(d_z, d_s)[None] / np.sqrt(d_s)
            Lam = np.random.randn(d_s, d_s)[None] / np.sqrt(d_s)

            ss_kernels.append(lib.GP.kernels.LEG(N, R, H, Lam))

        elif zc != "":
            raise ValueError("Invalid latent covariate type")

        tot_d_z += d_z
    
    if tot_d_z > 0:
        
        if len(ss_kernels) > 1:
            ss_kernels = lib.GP.kernels.StackMarkovian(ss_kernels)
        else:
            ss_kernels = ss_kernels[0]

        # site_init
        Tsteps = len(site_locs)
        site_obs = 0. * np.ones([Tsteps, tot_d_z, 1]) + 0*np.random.randn(Tsteps, tot_d_z, 1)
        site_Lcov = 1. * np.eye(tot_d_z)[None, ...].repeat(Tsteps, axis=0)

        ssgp = lib.GP.markovian.GaussianLTI(
            ss_kernels, site_locs, site_obs, site_Lcov, diagonal_site, fixed_grid_locs)
        
    else:
        ssgp = None
       
    # joint latent observed covariates
    lat_dims = list(range(d_x, d_x + tot_d_z))
    obs_dims = list(range(d_x))
    
    inputs_model = lib.inference.timeseries.GaussianLatentObservedSeries(
        ssgp, lat_dims, obs_dims, array_type=jnp.float32)



def setup_observations(
    config, covariates_kernel_dicts, covariates_induc_list, 
    ll_mode, filt_mode, map_mode, x_mode, z_mode, array_type
):
    """
    Assemble the encoding model
    
    :return:
        used covariates, inputs model, observation model
    """
    spktrain, cov, batch_info = data_tuple
    neurons, timesamples = spktrain.shape[0], spktrain.shape[-1]

    # checks
    if filt_mode == "" and config.filter_length > 0:
        raise ValueError
    
    ### kernel ###
    map_mode_comps = map_mode.split("-")
    num_induc = int(map_mode_comps[1])
    
    kernel_dicts, induc_list = [], []
    
    if map_mode == "nonparam_pp_gp":
        kernel_dicts_isi, induc_list_isi = ISI_kernel_dict_induc_list(ll_mode, num_induc)
        kernel_dicts += kernel_dicts_isi
        induc_list += induc_list_isi
        
    kernel_dicts += covariates_kernel_dicts
    induc_list += covariates_induc_list
    
    kernel_dicts_lat, induc_list_lat = latent_kernel(z_mode, num_induc, out_dims)
    kernel_dicts += kernel_dicts_lat
    induc_list += induc_list_lat
    
    kernel_list, dims_list = covariates_kernel_list(kernel_dicts, array_type)
    cov_kernel = lib.GP.kernels.Product(kernel_list, dims_list)
    induc_locs = np.array(induc_list)  # (f_dims, num_induc, x_dims)
    
    ### filter ###
    if filt_mode != "":
        obs_filter = build_spikefilters(filter_length)
    else:
        obs_filter = None
    
    ### GP observation model ###
    if map_mode == "factorized_gp":
        obs_model = build_factorized_gp(
            cov_kernel, induc_locs, obs_dims, obs_filter, jitter)

    elif map_mode == "rate_renewal_gp":
        obs_model = build_renewal_gp(
            cov_kernel, induc_locs, obs_filter, jitter, model_type="rescaled")
        
    elif map_mode == "mod_renewal_gp":
        obs_model = build_renewal_gp(
            cov_kernel, induc_locs, obs_filter, jitter, model_type="modulated")
        
    elif map_mode == "nonparam_pp_gp":
        obs_model = build_nonparametric(
            cov_kernel, induc_locs)
        
    else:
        raise ValueError('Invalid observation model type')
        
    return obs_model


def fit(parser_args, dataset_dict, observed_kernel_dict_induc_list, fix_param_names, save_name):
    """
    Fit and save the model
    General training loop
    """
    config = argparse.Namespace(**{**vars(parser_args), **dataset_dict["properties"]})
    
    # JAX
    if config.force_cpu:
        jax.config.update("jax_platform_name", "cpu")

    if config.double_arrays:
        array_type = jnp.float64
        jax.config.update("jax_enable_x64", True)
    else:
        array_type = jnp.float32

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.device}"
    
    # optimizer
    learning_rate_schedule = optax.exponential_decay(
        init_value=config.lr_start,
        transition_steps=batches,
        decay_rate=config.lr_decay,
        transition_begin=0,
        staircase=True,
        end_value=config.lr_end,
    )
    optim = optax.adam(learning_rate_schedule)
    
    # freeze parameters
    learn_mean = (ll_mode[0] != "U")
    
    select_fixed_params = lambda tree: [
        getattr(tree, name) for name in fix_param_names
    ]
    
    filter_spec = jax.tree_map(lambda _: True, model)
    filter_spec = eqx.tree_at(
        select_learnable_params,
        filter_spec,
        replace=(False,) * len(fix_param_names),
    )
    
    # inputs
    ISIs, covariates, inputs_model = build_inputs(dataset_dict, x_mode, z_mode, ll_mode)
    isi_order = ISIs.shape[-1] if ISIs is not None else 0
    
    # loss
    @partial(eqx.filter_value_and_grad, arg=filter_spec)
    def compute_loss(model, ic, inputs, targets):
        outputs = model(inputs, ic, dt, activity_L2 > 0.0)  # (time, tr, out_d)

        L2_weights = weight_L2 * ((model.W_rec) ** 2).sum() if weight_L2 > 0.0 else 0.0
        L2_activities = (
            activity_L2 * (outputs[1] ** 2).mean(1).sum() if activity_L2 > 0.0 else 0.0
        )
        return ((outputs[0] - targets) ** 2).mean(1).sum() + L2_weights + L2_activities

    @partial(eqx.filter_jit, device=jax.devices()[0])
    def make_step(model, ic, inputs, targets, opt_state):
        loss, grads = compute_loss(model, ic, inputs, targets)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        model = constraints(model)
        return loss, model, opt_state

    opt_state = optim.init(model)
    loss_tracker = []

    iterator = tqdm(range(epochs))  # graph iterations/epochs
    for ep in iterator:

        dataloader = iter(dataset)
        for (x, y) in dataloader:
            ic = jnp.zeros((x.shape[0], hidden_size))
            x = jnp.array(x.transpose(2, 0, 1))  # (time, tr, dims)
            y = jnp.array(y.transpose(2, 0, 1))

            if input_std > 0.0:
                x += input_std * jr.normal(prng_state, shape=x.shape)
                prng_state, _ = jr.split(prng_state)

            if priv_std > 0.0:
                eps = priv_std * jr.normal(
                    prng_state, shape=(*x.shape[:2], hidden_size)
                )
                prng_state, _ = jr.split(prng_state)
            else:
                eps = jnp.zeros((*x.shape[:2], hidden_size))

            loss, model, opt_state = make_step(model, ic, (x, eps), y, opt_state)
            loss = loss.item()
            loss_tracker.append(loss)

            loss_dict = {"loss": loss}
            iterator.set_postfix(**loss_dict)



    


    # training
    seeds = config.seeds
    
    dt = sample_bin * 1000 # ms
    start_t = first_all_spiked
    filter_length = 100

    timestamps = np.arange(tsteps - start_t) * dt


    observations = spktrain[:, start_t-filter_length:]
    covariates = np.stack([
        x_t, 
        y_t, 
        s_t, 
        hd_t, 
        theta_t, 
    ], axis=-1)[start_t:, ...]
    ISIs = lagged_ISIs[start_t:, ...].transpose(1, 0, 2)



    # fitting
    for seed in seeds:
        config.seed"] = seed
        print("seed: {}".format(seed))

        # seed everything
        seed = config.seed"]
        np.random.seed(seed)

        obs_dims = len(input_data)
        covariates_kernel_dicts, covariates_induc_list = observed_kernel_dict_induc_list()
        setup_observations(model_dict, covariates_kernel_dicts, covariates_induc_list)

        try:  # attempt to fit model
            full_model = setup_model(fitdata, model_dict, enc_used)
            full_model.to(dev)

            sch = lambda o: optim.lr_scheduler.MultiplicativeLR(
                o, lambda e: config.scheduler_factor
            )
            opt_tuple = (optim.Adam, config.scheduler_interval, sch)
            opt_lr_dict = {"default": config.lr}
            if z_mode == "T1":
                opt_lr_dict["mapping.kernel.kern1._lengthscale"] = config.lr_2
            for z_dim in full_model.input_group.latent_dims:
                opt_lr_dict[
                    "input_group.input_{}.variational.finv_std".format(z_dim)
                ] = config.lr_2

            full_model.set_optimizers(opt_tuple, opt_lr_dict)

            annealing = lambda x: 1.0
            losses = full_model.fit(
                config.max_epochs,
                loss_margin=config.loss_margin,
                margin_epochs=config.margin_epochs,
                kl_anneal_func=annealing,
                cov_samples=config.cov_MC,
                ll_samples=config.ll_MC,
                ll_mode=config.integral_mode,
            )

            # save and progress
            if os.path.exists(
                checkpoint_dir + model_name + "_result.p"
            ):  # check previous best losses
                with open(checkpoint_dir + model_name + "_result.p", "rb") as f:
                    results = pickle.load(f)
                    lowest_loss = results["training_loss"][-1]
            else:
                lowest_loss = np.inf  # nonconvex optimization, pick the best

            if losses[-1] < lowest_loss:

                ### save ###
                if not os.path.exists(config.checkpoint_dir):
                    os.makedirs(config.checkpoint_dir)

                savefile = config.checkpoint_dir + save_name
                savedata = {
                    "model": model,
                    "training_loss": loss_tracker,
                    "config": args,
                }
                pickle.dump(savedata, open(savefile, "wb"), pickle.HIGHEST_PROTOCOL)
                
                # save model
                torch.save(
                    {
                        "input_model": ,
                        "observation_model": obs_model, 
                        "training_loss": losses,
                        "seed": seed,
                        "config": config,
                    }, 
                    checkpoint_dir + model_name + ".pt",
                )

        except (ValueError, RuntimeError) as e:
            print(e)

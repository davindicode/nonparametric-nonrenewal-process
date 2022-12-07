import os
import sys

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from tqdm.autonotebook import tqdm

sys.path.append("..")

import pickle

import lib
import template


def gen_name():
    return


def main():
    ### parser ###
    parser = template.standard_parser(
        "%(prog)s [OPTION] [FILE]...", "Train recurrent network model."
    )

    parser.add_argument("--landmark_stage", default=0, type=int)
    parser.add_argument("--integration_stage", default=1000, type=int)
    parser.add_argument("--integration_g", default=1.0, type=float)
    parser.add_argument(
        "--stage_indicator", dest="stage_indicator", action="store_true"
    )
    parser.set_defaults(stage_indicator=False)
    parser.add_argument("--sigma_av", default=0.02, type=float)
    parser.add_argument("--mom_av", default=0.95, type=float)

    args = parser.parse_args()

    if args.force_cpu:
        jax.config.update("jax_platform_name", "cpu")

    if args.double_arrays:
        jax.config.update("jax_enable_x64", True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    ### setup ###
    seed = args.seed
    dt = args.dt
    epochs = args.epochs
    neurons = args.neurons
    priv_std = args.priv_std

    trials = args.trials

    ### task ###
    sigma_av = args.sigma_av
    mom_av = args.mom_av
    integration_g = args.integration_g

    trial_stages = {
        "landmark": args.landmark_stage,
        "integration": args.integration_stage,
    }

    inp, target = lib.tasks.angular_integration(
        trial_stages,
        trials,
        dt,
        sigma_av,
        mom_av,
        g=integration_g,
        indicator=args.stage_indicator,
        loaded_behaviour=None,
        seed=seed,
    )

    ### dataset ###
    batches = args.batches
    dataset = lib.utils.Dataset(inp, target, batches)

    learning_rate_schedule = optax.exponential_decay(
        init_value=args.lr_start,
        transition_steps=batches,
        decay_rate=args.lr_decay,
        transition_begin=0,
        staircase=True,
        end_value=args.lr_end,
    )
    optim = optax.adam(learning_rate_schedule)

    ### initialization ###
    np.random.seed(seed)
    in_size, hidden_size, out_size = inp.shape[1], neurons, target.shape[1]

    if args.dale:
        dale_column_sign = np.array(
            [1.0] * (hidden_size // 2) + [0.0] * (hidden_size // 2)
        )
    else:
        dale_column_sign = None

    if args.spiking:
        W_in = 0.5 * np.random.randn(hidden_size, in_size)
        W_in[:, -1] *= 10.0

        W_rec = 1.0 / np.sqrt(hidden_size) * np.random.randn(hidden_size, hidden_size)
        W_out = (
            10.0 / np.sqrt(hidden_size) * np.random.randn(out_size, hidden_size)
        )  # 1.
        bias = 0.0 * np.random.randn(hidden_size)
        out_bias = 0.0 * np.random.randn(out_size)
        ltau_v = 0.0 * np.random.randn(hidden_size) + np.log(20.0)
        ltau_s = 0.0 * np.random.randn(hidden_size) + np.log(20.0)
        ltau_I = 0.0 * np.random.randn(hidden_size) + np.log(20.0)

        v_thres = np.ones(hidden_size)
        v_reset = -0.3 * np.ones(hidden_size)

        model = lib.spiking.LIF_SNN(
            W_in, W_rec, W_out, bias, out_bias, ltau_v, ltau_s, ltau_I, v_thres, v_reset
        )
        constraints = lib.spiking.SNN_constraints(
            hidden_size, dale_column_sign=dale_column_sign, self_conn=False
        )

    else:
        W_in = 0.1 * np.random.randn(hidden_size, in_size)
        W_in[:, -1] *= 10.0

        W_rec = 1.0 / np.sqrt(hidden_size) * np.random.randn(hidden_size, hidden_size)
        W_out = 1.0 * np.random.randn(out_size, hidden_size)
        bias = 0.0 * np.random.randn(hidden_size)
        out_bias = 0.0 * np.random.randn(out_size)
        ltau_v = 0.0 * np.random.randn(hidden_size) + np.log(20.0)

        model = lib.analog.retanh_RNN(W_in, W_rec, W_out, bias, out_bias, ltau_v)
        constraints = lib.analog.RNN_constraints(
            hidden_size, dale_column_sign=dale_column_sign, self_conn=False
        )

    model = constraints(model)

    ### training ###
    template.train_model()


if __name__ == "__main__":
    main()


import sys

sys.path.append("..")
import jax, lib

import jax.numpy as jnp


def get_model(
    kernel,
    mapping,
    likelihood,
    x_dims,
    y_dims,
    tbin,
    obs_mc=20,
    lik_gh=20,
    seed=123,
    dtype=jnp.float32,
):
    # likelihood
    if likelihood == "Normal":
        var_y = 1.0 * jnp.ones(y_dims)
        lik = lib.likelihoods.Gaussian(y_dims, variance=var_y)
        f_dims = y_dims
    elif likelihood == "hNormal":
        lik = lib.likelihoods.HeteroscedasticGaussian(y_dims)  # , autodiff=False)
        f_dims = y_dims * 2
    elif likelihood == "Poisson":
        lik = lib.likelihoods.Poisson(y_dims, tbin)  # , autodiff=True)
        f_dims = y_dims
    lik.set_approx_integration(approx_int_method="GH", num_approx_pts=lik_gh)

    # mapping
    if mapping == "Id":
        x_dims = f_dims
        obs = lib.observations.Identity(f_dims, lik)
    elif mapping == "Lin":
        C = 0.1 * jax.random.normal(
            jax.random.PRNGKey(seed), shape=(f_dims, x_dims)
        )  # jnp.ones((f_dims, x_dims))
        # C = C/np.linalg.norm(C)
        b = 0.0 * jnp.ones((f_dims,))
        obs = lib.observations.Linear(lik, C, b)
    elif mapping == "bLin":
        mean_f = jnp.zeros(f_dims)
        scale_C = 1.0 * jnp.ones((1, x_dims))  # shared across f_dims
        blin_site_params = {
            "K_eta_mu": jnp.ones((f_dims, x_dims, 1)),
            # jax.random.normal(jax.random.PRNGKey(seed), shape=(f_dims, x_dims, 1)),
            "chol_K_prec_K": 1.0 * jnp.eye(x_dims)[None, ...].repeat(f_dims, axis=0),
        }
        obs = lib.observations.BayesianLinear(
            mean_f, scale_C, blin_site_params, lik, jitter=1e-5
        )
    elif mapping == "SVGP":
        len_fx = 1.0 * jnp.ones((f_dims, x_dims))  # GP lengthscale
        var_f = 1.0 * jnp.ones(f_dims)  # kernel variance
        kern = lib.kernels.SquaredExponential(
            f_dims, variance=var_f, lengthscale=len_fx
        )
        mean_f = jnp.zeros(f_dims)
        num_induc = 10
        induc_locs = jax.random.normal(
            jax.random.PRNGKey(seed), shape=(f_dims, num_induc, x_dims)
        )
        svgp_site_params = {
            "K_eta_mu": 1.0
            * jax.random.normal(jax.random.PRNGKey(seed), shape=(f_dims, num_induc, 1)),
            "chol_K_prec_K": 1.0 * jnp.eye(num_induc)[None, ...].repeat(f_dims, axis=0),
        }
        obs = lib.observations.SVGP(
            kern, mean_f, induc_locs, svgp_site_params, lik, jitter=1e-5
        )

    obs.set_approx_integration(approx_int_method="MC", num_approx_pts=obs_mc)

    # state space LDS
    if kernel == "Mat32":
        var_x = 1.0 * jnp.ones(x_dims)  # GP variance
        len_x = 10.0 * jnp.ones((x_dims, 1))  # GP lengthscale
        kernx = lib.kernels.Matern32(x_dims, variance=var_x, lengthscale=len_x)

    elif kernel == "LEG":
        N, R, B, Lam = lib.kernels.LEG.initialize_hyperparams(
            jax.random.PRNGKey(seed), 3, x_dims
        )
        kernx = lib.kernels.LEG(N, R, B, Lam)

    state_space = lib.latents.LinearDynamicalSystem(kernx, diagonal_site=True)

    model = lib.inference.CVI_SSGP(state_space, obs, dtype=dtype)
    name = (
        "{}x_{}y_{}ms_".format(x_dims, y_dims, int(tbin * 1000))
        + kernel
        + "_"
        + mapping
        + "_"
        + likelihood
    )
    return model, name


def split_params_func_GD(all_params, split_param_func=None):
    params, site_params = all_params["hyp"], all_params["sites"]

    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)

    # site params
    learned_sp = lib.utils.copy_pytree(site_params)
    fixed_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)

    return {"hyp": learned, "sites": learned_sp}, {"hyp": fixed, "sites": fixed_sp}


def split_params_func_NGDx(all_params, split_param_func=None):
    params, site_params = all_params["hyp"], all_params["sites"]

    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)

    # site params
    learned_sp = lib.utils.copy_pytree(site_params)
    fixed_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    for k in site_params["state_space"].keys():
        learned_sp["state_space"][k] = jnp.empty(0)
        fixed_sp["state_space"][k] = site_params["state_space"][k]

    return {"hyp": learned, "sites": learned_sp}, {"hyp": fixed, "sites": fixed_sp}


def split_params_func_NGDu(all_params, split_param_func=None):
    params, site_params = all_params["hyp"], all_params["sites"]

    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)

    # site params
    learned_sp = lib.utils.copy_pytree(site_params)
    fixed_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    for k in site_params["observation"].keys():
        learned_sp["observation"][k] = jnp.empty(0)
        fixed_sp["observation"][k] = site_params["observation"][k]

    return {"hyp": learned, "sites": learned_sp}, {"hyp": fixed, "sites": fixed_sp}


def split_params_func_NGD(all_params, split_param_func=None):
    params, site_params = all_params["hyp"], all_params["sites"]

    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)

    # site params
    learned_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    fixed_sp = lib.utils.copy_pytree(site_params)

    return {"hyp": learned, "sites": learned_sp}, {"hyp": fixed, "sites": fixed_sp}


def overwrite_GD(all_params, site_params):
    return all_params


def overwrite_NGDx(all_params, site_params):
    all_params["sites"]["state_space"] = site_params[
        "state_space"
    ]  # overwrite with NGDs
    return all_params


def overwrite_NGDu(all_params, site_params):
    all_params["sites"]["observation"] = site_params[
        "observation"
    ]  # overwrite with NGDs
    return all_params


def overwrite_NGD(all_params, site_params):
    all_params["sites"] = site_params  # overwrite with NGDs
    return all_params

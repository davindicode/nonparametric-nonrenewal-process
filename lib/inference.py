import math
import numpy as np

from functools import partial

import equinox as eqx

import jax
from jax import jit, lax, random, tree_map, value_and_grad, vmap

import jax.numpy as jnp
import jax.random as jr

from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve_triangular

from tqdm.autonotebook import tqdm

from .base import module

_log_twopi = math.log(2 * math.pi)



### inputs ###
def process_inputs(t, x_obs, y, dtype):
    """
    Order the inputs.
    :param t: training inputs (T,)
    :param x_obs: non-temporal coordinates (T, dims)
    :param y: observations at the training inputs (T, out)
    :return:
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ (steps-1,)
        t_train: training inputs (steps,)
        x_obs_train: observation inputs
        y_train: training observations (steps, N)
    """
    assert t.shape[0] == y.shape[0]  # matching number of time points
    ind = jnp.argsort(t, axis=0)

    t = jnp.array(t[ind, ...], dtype=dtype)
    y = jnp.array(y[ind, ...], dtype=dtype)
    dt = jnp.diff(t)

    if jnp.abs(jnp.diff(dt)).max() / dt.max() < 1e-2:  # uniform time grid
        dt = dt[:1]

    if x_obs is not None:
        if x_obs.ndim < 2:
            x_obs = x_obs.expand_dims(x_obs, 1)
        x_obs = jnp.array(x_obs[ind, ...], dtype=dtype)
    else:
        x_obs = None

    return t, dt, x_obs, y



### base ###
class ELBO_GPLVM(module):
    """
    The input-ouput (IO) mapping is deterministic or stochastic
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    """

    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.dtype = dtype

        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space

        self.y = None  # no training data set

    ### variational inference ###
    def filter_smoother(
        self, learned_all_params, fixed_all_params, prng_state, num_samps
    ):
        """
        Run Kalman filtering and RTS smoothing to obtain the ELBO
        """
        all_params = tree_map(
            lambda x, y: x if x is not None else y,
            learned_all_params,
            fixed_all_params,
            is_leaf=lambda x: x is None,
        )

        params, var_params = all_params["hyp"], all_params["sites"]
        jitter = self.jitter

        # parameters
        q_vh_ic = params["ic"]
        IF_params = params["IF_model"]

        ss_params, ss_var_params = params["state_space"], var_params["state_space"]
        eps_mapping, eps_var_params = params["eps_mapping"], var_params["eps_mapping"]
        mu_params, mu_var_params = params["mu_mapping"], var_params["mu_mapping"]
        lsigma_params, lsigma_var_params = (
            params["lsigma_mapping"],
            var_params["lsigma_mapping"],
        )

        # data
        t, dt, x_obs, y, mask = self.get_data()
        timedata = (t, dt)
        assert dt.shape[0] == 1  # assume uniform time intervals
        dt = dt[0]
        prng_keys = jr.split(prng_state, 4)

        ### E_q(x) [log p(x)/q(x)] and sample from posterior ###
        params, var_params = all_params["hyp"], all_params["sites"]

        x_samples, KL_ss = self.state_space.sample_posterior(
            ss_params,
            ss_var_params,
            prng_keys[0],
            num_samps,
            timedata,
            None,
            jitter,
            compute_KL=True,
        )  # (time, tr, x_dims, 1)

        eps_samples, KL_eps = self.eps_mapping.sample_posterior(
            eps_mapping, eps_var_params, prng_keys[1], x_samples, jitter, True
        )  # (time, tr, N)
        mu, KL_mu = self.mu_mapping.sample_posterior(
            mu_params, mu_var_params, prng_keys[2], x_obs, jitter, True
        )
        log_sigma, KL_lsigma = self.lsigma_mapping.sample_posterior(
            lsigma_params, lsigma_var_params, prng_keys[3], x_obs, jitter, True
        )

        KL = KL_ss + KL_eps + KL_mu + KL_lsigma  # mean(0) for MC/trials

        ### E_q(x) [log p(y|x)] ###
        I = mu + jnp.exp(log_sigma) * eps_samples
        ll = self.IF_model.Euler_fit(IF_params, dt, q_vh_ic, I, y)

        neg_ELBO = -ll + KL  # final objective
        return neg_ELBO

    def train_ELBO(
        self,
        all_params,
        prng_state,
        num_samps,
        split_all_params_func=None,
        take_grads=True,
    ):
        """
        Compute the full ELBO for hyperparameter learning
        E step is done in this function by updating site parameters
        M step given the site parameters before this E step is done by gradient descent with grads

        :param tuple inference_state: state for inference hyperparameters PRNG state, damping, jitter
        :param lambda split_all_params_func: function to split into learnable and fixed parameters
        :param bool take_grads: if True, use autodiff to get hyperparameter gradients. Running pure
                                E steps involves setting this to False
        :return:
            objective: the negative ELBO -E_q(x)[ ELBO(x) ]
            grads: the derivative of the objective w.r.t. the model hyperparameters
            inference_state: state list for inference settings
        """
        if split_all_params_func is None:
            learned_all_params = all_params
            fixed_all_params = tree_map(lambda prms: None, all_params)
        else:
            learned_all_params, fixed_all_params = split_all_params_func(all_params)

        # filtering-smoothing
        if take_grads:  # compute ELBO and gradients via autodiff
            objective, grads = value_and_grad(self.filter_smoother, argnums=0)(
                learned_all_params, fixed_all_params, prng_state, num_samps
            )
            grads = tree_map(
                lambda grds, prms: jnp.zeros_like(prms) if grds is None else grds,
                grads,
                all_params,
                is_leaf=lambda x: x is None,
            )

        else:
            objective, aux = self.filter_smoother(
                learned_all_params, fixed_all_params, prng_state, num_samps
            )
            grads = None

        return objective, grads
    
    
    def train_grads(
        model,
        constraints,
        select_learnable_params,
        dataset,
        optim,
        dt,
        epochs,
        in_size,
        hidden_size,
        out_size,
        weight_L2,
        activity_L2,
        prng_state,
        priv_std,
        input_std,
    ):
        # freeze parameters
        filter_spec = jax.tree_map(lambda _: False, model)
        filter_spec = eqx.tree_at(
            select_learnable_params,
            filter_spec,
            replace=(True,) * len(select_learnable_params(model)),
        )

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

        return model, loss_tracker
    

    ### sample ###
    def sample_prior(self, prng_state, num_samps, x_obs=None, timedata=None):
        """
        Sample from the generative model
        """
        params = self.get_all_params()["hyp"]
        jitter = self.jitter

        if x_obs is None:
            x_obs = self.x_obs

        if timedata is None:  # default to training data
            t, dt = self.t, self.dt
        else:
            t, dt = timedata
        assert dt.shape[0] == 1  # assume uniform time intervals

        x_samples = self.state_space.sample_prior(
            params["state_space"], prng_state, num_samps, (t, dt), jitter
        )  # (time, tr, x_dims, 1)

        y, q_vh, I, eps_samples = self.simulate_spiketrains(
            params, None, prng_state, dt[0], x_samples, x_obs, jitter, prior=True
        )
        return y, q_vh, I, eps_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        all_params = self.get_all_params()
        params, var_params = all_params["hyp"], all_params["sites"]
        jitter = self.jitter

        timedata = (self.t, self.dt)
        assert self.dt.shape[0] == 1  # assume uniform time intervals
        prng_keys = jr.split(prng_state, 2)

        x_samples = self.state_space.sample_posterior(
            params["state_space"],
            var_params["state_space"],
            prng_keys[0],
            num_samps,
            timedata,
            None,
            jitter,
            compute_KL=False,
        )[
            0
        ]  # (time, tr, x_dims, 1)
        y, q_vh, I, eps_samples = self.simulate_spiketrains(
            params,
            var_params,
            prng_keys[1],
            self.dt[0],
            x_samples,
            self.x_obs,
            jitter,
            prior=False,
        )
        return y, q_vh, I, eps_samples

    
    ### evaluation ###
    def evaluate_metric(self):
        return
    
    
    
class ELBO_SwitchingSSGP(module):
    """
    The input-ouput (IO) mapping is deterministic or stochastic
    Switching prior
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    """

    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.dtype = dtype

        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space

        self.y = None  # no training data set
        
        
        
class ELBO_DTGPSSM(module):
    """
    GPSSM ELBO
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    """

    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.dtype = dtype

        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space

        self.y = None  # no training data set
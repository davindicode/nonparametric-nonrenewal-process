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



def build_factorized():
    
    return



def build_



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


def train_model():
    select_learnable_params = lambda tree: [
        getattr(tree, name) for name in args.learn_params
    ]

    if args.spiking:
        model, loss_tracker = lib.tasks.train_spiking(
            model,
            constraints,
            select_learnable_params,
            dataset,
            optim,
            args.beta,
            args.surrogate_type,
            args.ignore_reset_grad,
            dt,
            epochs=epochs,
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            weight_L2=args.weight_L2,
            activity_L2=args.activity_L2,
            prng_state=jr.PRNGKey(seed),
            priv_std=priv_std,
            input_std=args.input_std,
        )
    else:
        model, loss_tracker = lib.tasks.train_analog(
            model,
            constraints,
            select_learnable_params,
            dataset,
            optim,
            dt,
            epochs=epochs,
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            weight_L2=args.weight_L2,
            activity_L2=args.activity_L2,
            prng_state=jr.PRNGKey(seed),
            priv_std=priv_std,
            input_std=args.input_std,
        )

    ### save ###
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    savefile = args.checkpoint_dir + args.model_name
    savedata = {
        "model": model,
        "training_loss": loss_tracker,
        "config": args,
    }
    pickle.dump(savedata, open(savefile, "wb"), pickle.HIGHEST_PROTOCOL)


### training ###
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


def train_analog(
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


def train_spiking(
    model,
    constraints,
    select_learnable_params,
    dataset,
    optim,
    beta,
    surrogate_type,
    ignore_reset_grad,
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
        outputs = model(
            inputs, ic, dt, activity_L2 > 0.0, beta, surrogate_type, ignore_reset_grad
        )  # (time, tr, out_d)

        L2_weights = weight_L2 * ((model.W_rec) ** 2).sum() if weight_L2 > 0.0 else 0.0
        L2_activities = (
            activity_L2 * (outputs[2][..., -2] ** 2).mean(1).sum()
            if activity_L2 > 0.0
            else 0.0
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

    iterator = tqdm(range(epochs))
    for ep in iterator:

        dataloader = iter(dataset)
        for (x, y) in dataloader:
            ic = jnp.zeros((x.shape[0], hidden_size, model.state_d))
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


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append("../../neuroppl/")
sys.path.append("../lib/")


import neuroppl as nppl
from neuroppl import utils


import lib

import argparse
import pickle



### definitions
max_Horder = 5





def gen_smooth_trajectories(Tl, dt, trials, tau_list):
    """
    generate smooth GP input
    """
    tau_list_ = tau_list*trials

    out_dims = len(tau_list_)
    l = np.array(tau_list_)[None, :]
    v = np.ones(out_dims)
    kernel_tuples = [('variance', v), 
                     ('RBF', 'euclid', l)]

    with torch.no_grad():
        kernel, _, _ = lib.models.create_kernel(kernel_tuples, 'softplus', torch.double)

        T = torch.arange(Tl)[None, None, :, None]*dt
        K = kernel(T, T)[0, ...]
        K.view(out_dims, -1)[:, ::Tl+1] += 1e-6

    L = torch.cholesky(K)
    eps = torch.randn(out_dims, Tl, 1).double()
    v = (L @ eps)[..., 0]
    a_t = v.data.numpy().reshape(trials, -1, Tl)
    return a_t # trials, tau_arr, time




def gen_torus_RW(Tl, trials, dt, tau_list):
    """
    Torus random walk
    """
    tau_arr = np.array(tau_list)
    tn = len(tau_arr)
    hd_t = np.empty((trials, tn, Tl))

    hd_t[..., 0] = np.random.rand(trials, tn)*2*np.pi
    rn = np.random.randn(trials, tn, Tl)/np.sqrt(dt)
    for k in range(1, Tl):
        hd_t[..., k] = hd_t[..., k-1] + rn[..., k]*dt/tau_arr[None, :]

    hd_t = hd_t % (2*np.pi)
    return hd_t # trials, tau_arr, time





### models
def hdc_pop(track_samples, covariates, neurons, trials=1):
    """
    Gamma renewal process bumps
    """
    # Von Mises fields
    angle_0 = np.linspace(0, 2*np.pi, neurons+1)[:-1]
    beta = np.random.rand(neurons)*2.0 + 0.5
    rate_0 = np.random.rand(neurons)*4.0+4.0
    w = np.stack([np.log(rate_0), beta*np.cos(angle_0), beta*np.sin(angle_0)]).T # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_rate = rate_lib.models.vonMises_GLM(neurons)
    vm_rate.set_params(torch.from_numpy(w))
    
    return vm_rate






def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...", description="Generate synthetic data."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('--dataset', type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    dev = utils.pytorch.get_device(gpu=args.gpu)
    dataset = args.dataset
    
    ### generation ###
    if dataset == 0: # HDC
        dt = 0.001 # s
        track_samples = 100000
        trials = 1
        neurons = 50
        tau_list = [1.0]

        hd_t = gen_torus_RW(track_samples, trials, dt, tau_list)[:, 0, :]

        # renewal gamma
        covariates = [torch.from_numpy(hd_t)[..., None]]
        mapping = hdc_pop(track_samples, covariates, neurons, trials=trials)
        mapping.to(dev)

        # Gamma process output
        shape = torch.exp(-1.+3.*torch.rand(neurons))
        likelihood = nppl.likelihoods.Gamma(dt, neurons, 'exp', shape, allow_duplicate=False)

        # sample and convert
        syn_train = lib.helper.sample_Y(mapping, likelihood, covariates, trials=1, MC=1)

        syn_inds = []
        for train_ in syn_train:
            syn_inds_ = []
            for train__ in train_:
                syn_inds_.append(utils.neural.binned_to_indices(train__))

            syn_inds.append(syn_inds_)

        # generate past ISIs
        hist_len = 1
        trial = 0

        rcov = [hd_t[0, :]]
        cov = lib.lib.helper.pp_covariates(rcov, syn_inds[trial], max_Horder, hist_len, dt).numpy()
        rc_t = syn_train[0, ...]

        np.savez_compressed('../data/HDC_gamma', spktrain=rc_t, cov=cov, shape=shape)
        torch.save({'model': mapping.state_dict()}, '../analysis/saves/HDC_gamma_model')

    elif dataset == 1: # Place cells
        dt = 0.001 # s
        track_samples = 30000
        trials = 1
        neurons = 50
        
    elif dataset == 2: # Izhikevich
        dt = 0.001 # s
        track_samples = 30000
        trials = 10
        neurons = 1
        
        I_t = 0
    

    
    
if __name__ == "__main__":
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import subprocess
import os
import argparse



import pickle

import sys
sys.path.append("../../neuroppl/")
sys.path.append("..")

import neuroppl as nppl
from neuroppl import utils

import lib

from init_data import max_Horder




def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run diode simulations."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    
    parser.add_argument('--dataset', action='store', type=str)
    
    parser.add_argument('--batchsize', default=10000, type=int)
    parser.add_argument('--modes', nargs='+', type=int)
    parser.add_argument('--cv', nargs='+', type=int)
    parser.add_argument('--cv_folds', default=5, type=int)
    
    parser.add_argument('--ncvx', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cov_MC', default=1, type=int)
    parser.add_argument('--ll_MC', default=10, type=int)
    
    parser.add_argument('--jitter', default=1e-5, type=float)
    parser.add_argument('--maxiters', default=3000, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_2', default=1e-3, type=float)
    
    parser.add_argument('--likelihood', action='store', type=str)
    parser.add_argument('--x_mode', default='', action='store', type=str)
    parser.add_argument('--z_mode', default='', action='store', type=str)
    parser.add_argument('--num_induc', type=int)
    parser.add_argument('--delays', nargs='+', default=[0], type=int)
    
    args = parser.parse_args()
    return args








def get_dataset(dataset):
    data = np.load('../data/hc5_{}_ext.npz'.format(dataset))
    cov = data['covariates'] # time, dims, neurons
    cov_tuple = tuple(np.transpose(cov, (1, 2, 0))[..., None])
    
    timesamples = cov.shape[0]
    tbin = 0.001

    data = np.load('../data/hc5_{}.npz'.format(dataset))
    spktrain = data['spktrain'] # full
    arena = data['arena']
    
    bin_size = 1 # resolution of data provided
    max_count = 1
    spktrain[spktrain > 1.] = 1.
    units_used = spktrain.shape[-2]
    
    model_name = 'CA{}_'.format(dataset)
    metainfo = (arena,)
    return cov_tuple, units_used, tbin, timesamples, spktrain, max_count, bin_size, metainfo, model_name



### model
def inputs_used(x_mode, z_mode, cov_tuple, batch_info, tensor_type):
    """
    Create the used covariates list.
    """
    timesamples = cov_tuple[0].shape[0]
    b = [torch.from_numpy(b) for b in cov_tuple]
    x_t, y_t, s_t, th_t, hd_t, w_t, time_t, isi1, isi2, isi3, isi4, isi5 = b
    
    # x
    if x_mode[:-1] == 'th-pos-isi':
        H_isi = int(x_mode[-1])
        input_data = [th_t, x_t, y_t]
        for h in range(-max_Horder, -max_Horder+H_isi):
            input_data += [b[h]]
            
    elif x_mode == 'th-pos':
        input_data = [th_t, x_t, y_t]
            
    elif x_mode[:-1] == 'th-hd-pos-s-t-isi':
        input_data = [th_t, hd_t, x_t, y_t, s_t, time_t]
    
    elif x_mode is None:
        input_data = []
        
    else:
        raise ValueError
        
    d_x = len(input_data)
        
    latents, d_z = lib.helper.latent_objects(z_mode, d_x, timesamples, tensor_type)
    input_data += latents
    return input_data, d_x, d_z



def enc_used(map_mode, ll_mode, x_mode, z_mode, cov_tuple, in_dims, inner_dims, jitter, tensor_type):
    """
    """
    def kernel_used(x_mode, z_mode, cov_tuple, num_induc, out_dims, var, tensor_type):
        """
        Get kernel and inducing points
        """
        x_t, y_t, s_t, th_t, hd_t, w_t, time_t, isi1, isi2, isi3, isi4, isi5 = cov_tuple

        left_x = x_t.min()
        right_x = x_t.max()
        bottom_y = y_t.min()
        top_y = y_t.max()

        v = var*torch.ones(out_dims)

        l = 10.*np.ones(out_dims)
        l_ang = 5.*np.ones(out_dims)
        l_s = 10.*np.ones(out_dims)
        l_time = time_t.max()/2.*np.ones(out_dims)
        l_one = np.ones(out_dims)

        upperT = 2.0 # e^2
        l_ISI = 3.*np.ones(out_dims)
        tau_0 = 5.*np.ones((out_dims, max_Horder))
        #tau_0 = np.empty((out_dims, max_Horder))
        #for k in range(1, 1+max_Horder):
        #    tau_0[:, -k] = cov_tuple[-k].mean(1)[:, 0]

        # x
        if x_mode[:-1] == 'th-pos-isi':
            H_isi = int(x_mode[-1])
            ind_list = [
                np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                np.random.uniform(left_x, right_x, size=(num_induc,)), 
                np.random.uniform(bottom_y, top_y, size=(num_induc,))
            ]
            for h in range(H_isi):
                ind_list += [np.random.uniform(0.0, tau_0[:, h].mean(), size=(num_induc,))]

            kernel_tuples = [
                ('variance', v), 
                ('SE', 'torus', torch.tensor([l_ang])), 
                ('SE', 'euclid', torch.tensor([l, l])), 
                ('tSE', 'euclid', torch.tensor([l_ISI]*H_isi), tau_0[:, :H_isi])
            ]
            
        elif x_mode == 'th-pos':
            ind_list = [
                np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                np.random.uniform(left_x, right_x, size=(num_induc,)), 
                np.random.uniform(bottom_y, top_y, size=(num_induc,))
            ]

            kernel_tuples = [
                ('variance', v), 
                ('SE', 'torus', torch.tensor([l_ang])), 
                ('SE', 'euclid', torch.tensor([l, l]))
            ]

        elif x_mode[:-1] == 'th-hd-pos-s-t-isi':
            ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                        np.random.rand(num_induc)*2*np.pi, 
                        np.random.uniform(left_x, right_x, size=(num_induc,)), 
                        np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                        np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                        np.linspace(0, time_t.max(), num_induc)]
            kernel_tuples = [
                ('variance', v), 
                ('SE', 'torus', torch.tensor([l_ang, l_ang])), 
                ('SE', 'euclid', torch.tensor([l, l])), 
                ('SE', 'euclid', torch.tensor([l_s])), 
                ('SE', 'euclid', torch.tensor([l_time]))
            ]

        elif x_mode == '':
            ind_list = []
            kernel_tuples = [('variance', v)]

        else:
            raise ValueError

        # z
        latent_k, latent_u = lib.helper.latent_kernel(z_mode)
        kernel_tuples += latent_k
        ind_list += latent_u

        # objects
        kernelobj, constraints = lib.helper.create_kernel(kernel_tuples, 'exp', tensor_type)

        Xu = torch.tensor(ind_list).T[None, ...].repeat(out_dims, 1, 1)
        inpd = Xu.shape[-1]
        inducing_points = nppl.kernels.kernel.inducing_points(out_dims, Xu, constraints)

        return kernelobj, inducing_points

    if map_mode[:4] == 'svgp':
        num_induc = int(map_mode[4:])

        mean = 0 if ll_mode == 'U' else torch.zeros((inner_dims)) # not learnable vs learnable
        learn_mean = (ll_mode != 'U')

        var = 1.0 # initial kernel variance
        kernelobj, inducing_points = kernel_used(x_mode, z_mode, cov_tuple, num_induc, 
                                                  inner_dims, var, tensor_type)

        mapping = nppl.mappings.SVGP(
            in_dims, inner_dims, kernelobj, inducing_points=inducing_points, 
            jitter=jitter, whiten=True, 
            mean=mean, learn_mean=learn_mean, 
            tensor_type=tensor_type
        )
    
        # heteroscedastic likelihoods
        if ll_mode[0] == 'h':
            kernelobj, inducing_points = kernel_used(x_mode, z_mode, cov_tuple, num_induc, 
                                                      inner_dims, var, tensor_type)
            
            hgp = nppl.mappings.SVGP(
                in_dims, inner_dims, kernelobj, inducing_points=inducing_points, 
                jitter=jitter, whiten=True, 
                mean=torch.zeros((inner_dims)), learn_mean=True, 
                tensor_type=tensor_type
            )
            
        else:
            hgp = None
        
    else:
        raise ValueError
        
    return mapping, hgp





### main
def main():
    parser = lib.models.standard_parser("%(prog)s [OPTION] [FILE]...", "Fit model to data.")
    parser.add_argument('--dataset', action='store', type=str)
    args = parser.parse_args()
    
    dev = utils.pytorch.get_device(gpu=args.gpu)
    
    dataset = args.dataset
    dataset_tuple = get_dataset(dataset)

    lib.models.training(dev, args, dataset_tuple, inputs_used, enc_used)

                


if __name__ == "__main__":
    main()
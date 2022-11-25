import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


import pickle

import sys
sys.path.append("../../neuroppl/")
sys.path.append("../lib/")


import neuroppl as nppl
from neuroppl import utils

import lib



### data
def get_dataset(session_id, phase, bin_size, single_spikes=False):

    data = np.load('../data/Mouse{}_{}.npz'.format(session_id, phase))
    spktrain = data['spktrain']
    x_t = data['x_t']
    y_t = data['y_t']
    hd_t = data['hd_t']
    region_edge = data['region_edge']
    #arena = data['arena']

    sample_bin = 0.001

    neurons = spktrain.shape[0]
    track_samples = spktrain.shape[1]

    tbin, resamples, rc_t, (rhd_t, rx_t, ry_t) = utils.neural.bin_data(
        bin_size, sample_bin, spktrain, track_samples, 
        (np.unwrap(hd_t), x_t, y_t), average_behav=True, binned=True
    )

    # recompute velocities
    rw_t = (rhd_t[1:]-rhd_t[:-1])/tbin
    rw_t = np.concatenate((rw_t, rw_t[-1:]))

    rvx_t = (rx_t[1:]-rx_t[:-1])/tbin
    rvy_t = (ry_t[1:]-ry_t[:-1])/tbin
    rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
    rs_t = np.concatenate((rs_t, rs_t[-1:]))
    rtime_t = np.arange(resamples)*tbin

    units_used = rc_t.shape[0]
    rcov = (utils.signal.WrapPi(rhd_t, True), rw_t, rs_t, rx_t, ry_t, rtime_t)
    
    if single_spikes is True:
        rc_t[rc_t > 1.] = 1.
    
    max_count = int(rc_t.max())
    model_name = 'HDC{}{}_'.format(session_id, phase)
    metainfo = (region_edge,)
    return rcov, units_used, tbin, resamples, rc_t, max_count, bin_size, metainfo, model_name





### model
def inputs_used(x_mode, z_mode, cov_tuple, behav_info, tensor_type):
    """
    Create the used covariates list.
    """
    timesamples = cov_tuple[0].shape[0]
    tb = [torch.from_numpy(b) for b in cov_tuple]
    hd_t, w_t, s_t, x_t, y_t, time_t = tb
    batch_info, out_dims, behav_series = behav_info
    bb = [torch.from_numpy(b) for b in behav_series]
    hd_b, w_b, s_b, x_b, y_b, time_b = bb
    
    # x
    if x_mode == 'hd-w-s-pos-t':
        input_data = [hd_t, w_t, s_t, x_t, y_t, time_t]
        
    elif x_mode[:4] == 'hdWP':
        induc_num = int(x_mode[4:])
        input_group = nppl.inference.input_group(tensor_type)
        input_group.set_XZ([time_t], timesamples, batch_info=batch_info)
    
        input_data = [
            nppl.inference.probabilistic_mapping(input_group, 
                                                 lib.warping.monotonic_GP_flows(out_dims, induc_num, time_b, hd_b))
        ]

    elif x_mode == '':
        input_data = []
        
    else:
        raise ValueError
        
    d_x = len(input_data)
    
    latents, d_z = lib.helper.latent_objects(z_mode, d_x, timesamples, tensor_type)
    input_data += latents
    return input_data, d_x, d_z






def enc_used(map_mode, ll_mode, x_mode, z_mode, behav_tuple, in_dims, inner_dims, jitter, tensor_type):
    """
    """
    def kernel_used(x_mode, z_mode, behav_tuple, num_induc, out_dims, var, tensor_type):
        hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple

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
        l_w = w_t.std()*np.ones(out_dims)

        # x
        if x_mode == 'hd-w-s-pos-t':
            ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                        np.random.randn(num_induc)*w_t.std(), 
                        np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                        np.random.uniform(left_x, right_x, size=(num_induc,)), 
                        np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                        np.linspace(0, time_t.max(), num_induc)]
            kernel_tuples = [('variance', v), 
                  ('RBF', 'torus', torch.tensor([l_ang])), 
                  ('RBF', 'euclid', torch.tensor([l_w, l_s, l, l, l_time]))]

        elif x_mode == 'hd' or x_mode[:4] == 'hdWP':
            ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1]]
            kernel_tuples = [('variance', v), 
                  ('RBF', 'torus', torch.tensor([l_ang]))]

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
        inducing_points = nppl.kernels.kernel.inducing_points(out_dims, Xu, constraints, 
                                                              tensor_type=tensor_type)

        return kernelobj, inducing_points


    if map_mode[:4] == 'svgp':
        num_induc = int(map_mode[4:])

        mean = 0 if ll_mode == 'U' else torch.zeros((inner_dims)) # not learnable vs learnable
        learn_mean = (ll_mode != 'U')

        var = 1.0 # initial kernel variance
        kernelobj, inducing_points = kernel_used(x_mode, z_mode, behav_tuple, num_induc, 
                                                 inner_dims, var, tensor_type)

        mapping = nppl.mappings.SVGP(
            in_dims, inner_dims, kernelobj, inducing_points=inducing_points, 
            jitter=jitter, whiten=True, 
            mean=mean, learn_mean=learn_mean, 
            tensor_type=tensor_type
        )
    
        # heteroscedastic likelihoods
        if ll_mode[0] == 'h':
            kernelobj, inducing_points = kernel_used_(datatype, x_mode, z_mode, behav_tuple, num_induc, 
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
    parser.add_argument('--session_id', action='store', type=str)
    parser.add_argument('--phase', action='store', type=str)
    args = parser.parse_args()
    
    dev = utils.pytorch.get_device(gpu=args.gpu)
    
    session_id = args.session_id# = ['Mouse12-120806', 'Mouse28-140313']
    phase = args.phase#= ['sleep', 'wake']
    dataset_tuple = get_dataset(session_id, phase, args.bin_size, args.single_spikes)

    lib.models.training(dev, args, dataset_tuple, inputs_used, enc_used)

                


if __name__ == "__main__":
    main()
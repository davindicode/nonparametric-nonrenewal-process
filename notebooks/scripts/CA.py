import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import subprocess
import os
import argparse



import pickle

import sys
sys.path.append("../../neuroprob")

import os


import neuroprob as mdl
from neuroprob import utils

import models
import model_utils





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
    parser.add_argument('--binsize', default=20, type=int)
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




def get_dataset(dataset, bin_size):
    
    data = np.load('../data/hc5_{}.npz'.format(dataset))
    sample_bin = 0.001
        
    #elif data_type == 1:
    #    dataset = 'hc3'
    #    session_id = 'ec014.468'
    #    data = np.load('./data/{}_{}_reduced.npz'.format(dataset, session_id))
    #    sample_bin = data['sample_bin']
        
    spktrain = data['spktrain']
    x_t = data['x_t']
    y_t = data['y_t']
    hd_t = data['hd_t']
    theta_t = data['theta_t']
    arena = data['arena']
        
    neurons = spktrain.shape[0]
    track_samples = spktrain.shape[1]
        
    tbin, resamples, rc_t, (rx_t, ry_t, rtheta_t, rhd_t) = utils.neural.bin_data(
        bin_size, sample_bin, spktrain, track_samples, 
        (x_t, y_t, np.unwrap(theta_t), np.unwrap(hd_t)), average_behav=True, binned=True
    )
    
    # recompute velocities
    rw_t = (rhd_t[1:]-rhd_t[:-1])/tbin
    rw_t = np.concatenate((rw_t, rw_t[-1:]))

    rvx_t = (rx_t[1:]-rx_t[:-1])/tbin
    rvy_t = (ry_t[1:]-ry_t[:-1])/tbin
    rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
    rs_t = np.concatenate((rs_t, rs_t[-1:]))
    rtime_t = np.arange(resamples)*tbin
    
    rhd_t = utils.signal.WrapPi(rhd_t, True)
    rtheta_t = utils.signal.WrapPi(rtheta_t, True)

    rcov = (rx_t, ry_t, rs_t, rtheta_t, rhd_t, rw_t, rtime_t)
    
    max_count = int(rc_t.max())
    model_name = 'CA{}_'.format(dataset)
    metainfo = (arena,)
    return rcov, neurons, tbin, resamples, rc_t, max_count, metainfo, model_name





### model
def cov_used(x_mode, z_mode, behav_tuple):
    """
    Create the used covariates list.
    """
    timesamples = behav_tuple[0].shape[0]
    x_t, y_t, s_t, th_t, hd_t, w_t, time_t = behav_tuple
    d_z = 0
    
    # x
    if x_mode == 'th-hd-pos-s-t':
        covariates = [th_t, hd_t, x_t, y_t, s_t, time_t]
    
    elif x_mode == '':
        covariates = []
        
    else:
        raise ValueError
        
    # z
    z_dims = []
    if z_mode[:1] == 'R':
        d_z = int(z_mode[1:])
        covariates += [[np.random.randn(timesamples, d_z)*0.1, np.ones((timesamples, d_z))*0.01]]
        z_dims.append(d_x)
        
    d_x = len(covariates)-d_z
    return covariates, d_x, d_z, z_dims



def kernel_used(x_mode, z_mode, behav_tuple, num_induc, outdims, var=1.):
    """
    Get kernel and inducing points
    """
    x_t, y_t, s_t, th_t, hd_t, w_t, time_t = behav_tuple
    
    left_x = x_t.min()
    right_x = x_t.max()
    bottom_y = y_t.min()
    top_y = y_t.max()
    
    l = 10.*np.ones(outdims)
    l_ang = 5.*np.ones(outdims)
    l_s = 10.*np.ones(outdims)
    v = var*np.ones(outdims)
    l_time = time_t.max()/2.*np.ones(outdims)
    l_one = np.ones(outdims)
    
    # x
    if x_mode == 'th-hd-pos-s-t':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.rand(num_induc)*2*np.pi, 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc)]
        kernel_tuples = [
            ('variance', v), 
            ('RBF', 'torus', np.array([l_ang, l_ang])), 
            ('RBF', 'euclid', np.array([l, l])), 
            ('RBF', 'euclid', np.array([l_s])), 
            ('RBF', 'euclid', np.array([l_time]))
        ]
        
    elif x_mode is None: # for GP filters
        ind_list = []
        kernel_tuples = [('variance', v)]

    else:
        raise ValueError
        
    # z
    if z_mode[:1] == 'R':
        dz = int(z_mode[1:]) 
        for h in range(dz):
            ind_list += [np.random.randn(num_induc)]
        kernel_tuples += [('RBF', 'euclid', np.array([l_one]*dz))]
        
    elif z_mode != '':
        raise ValueError
        
    return kernel_tuples, ind_list



def get_VI_blocks(x_mode, z_mode, d_x, meta_data, mdl_batchsize):
    """
    """
    # x
    if x_mode == 'th-hd-pos-s-t':
        VI_tuples = [(None, None, None, 1)]*d_x
        
    elif x_mode is None:
        VI_tuples = []

    else:
        raise ValueError
        
    # z
    if z_mode == 'R1':
        VI_tuples += [(['RW', (4.0, 1.0, True, False)], 'Normal', 'euclid', 1)]

    elif z_mode[:1] == 'R':
        d_z = int(z_mode[1:])
        VI_tuples += [(['RW', (np.array([4.0]*d_z), np.array([1.0]*d_z), True, False)], 'Normal', 'euclid', d_z)]
        
    in_dims = sum([d[3] for d in VI_tuples])
    return VI_tuples, in_dims, mdl_batchsize








### main
def main():
    parser = init_argparse()
    dev = utils.pytorch.get_device(gpu=parser.gpu)
    
    dataset_tuple = get_dataset(parser.dataset, parser.binsize)

    nonconvex_trials = parser.ncvx
    folds = parser.cv_folds
    cv_runs = parser.cv
    
    # mode
    ll_mode = parser.likelihood
    x_mode = parser.x_mode
    z_mode = parser.z_mode
    num_induc = parser.num_induc
    delays = parser.delays
    m = (ll_mode, x_mode, z_mode, num_induc, delays)
    
        
    ### model
    rcov, units_used, tbin, resamples, rc_t, max_count, metainfo, mdl_name = dataset_tuple
    model_structure, mdl_batchsize = models.setup_model(dataset_tuple, m, metainfo, parser.batchsize, 
                                                        cov_used, get_VI_blocks, kernel_used)


    ### training loop
    z_dims = model_structure[-2]
    for cvdata in model_utils.preprocess_data(z_dims, folds, delays, cv_runs, mdl_batchsize, rc_t, resamples, rcov):
        kcv_str, ftrain, fcov, vtrain, vcov, batch_size = cvdata
        data = (tbin, cov_used(x_mode, z_mode, fcov)[0], max_count, ftrain)

        lowest_loss = np.inf # nonconvex pick the best
        for kk in range(nonconvex_trials):

            ### model fitting
            retries = 0
            while True:
                try:
                    full_model = models.get_model(data, model_structure, \
                                                  jitter=parser.jitter, batch_size=batch_size, 
                                                  hist_len=1, filter_props=None, J=100)
                    full_model.to(dev)

                    # fit
                    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
                    opt_tuple = (optim.Adam, 100, sch)
                    opt_lr_dict = {'default': parser.lr}
                    for z_dim in z_dims:
                        opt_lr_dict['inputs.lv_std_{}'.format(z_dim)] = parser.lr_2

                    full_model.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

                    annealing = lambda x: 1.0#min(1.0, 0.002*x)

                    losses = full_model.fit(parser.maxiters, loss_margin=-1e0, margin_epochs=100, kl_anneal_func=annealing, 
                                            cov_samples=parser.cov_MC, ll_samples=parser.ll_MC)
                    break

                except (RuntimeError, AssertionError):
                    print('Retrying...')
                    if retries == 3: # max retries
                        print('Stopped after max retries.')
                        raise ValueError
                    retries += 1

            ### progress and save
            if losses[-1] < lowest_loss:
                lowest_loss = losses[-1]

                # save model        
                model_name = models.gen_name(
                    mdl_name, ll_mode, x_mode, z_mode, binsize, max_count, folds, kcv_str
                )

                if not os.path.exists('./checkpoint'):
                    os.makedirs('./checkpoint')
                torch.save({'full_model': full_model.state_dict()}, './checkpoint/' + model_name)

                

if __name__ == "__main__":
    main()
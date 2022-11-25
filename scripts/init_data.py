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



# definitions
max_Horder = 5 # renewal if 1
datasets = []



# functions
def HDC_dataset(session_id, phase, bin_size):

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
    
    max_count = int(rc_t.max())
    model_name = 'HDC{}{}_'.format(session_id, phase)
    metainfo = (region_edge,)
    return rcov, units_used, tbin, resamples, rc_t, max_count, metainfo, model_name



def CA_dataset(dataset, bin_size):
    
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



### main
def main():
    # load HDC dataset
    session_id = '12-120806'
    phase = 'wake'
    rcov, neurons, tbin, resamples, rc_t, max_count, metainfo, model_name = HDC_dataset(session_id, phase, 1)

    filename = 'Mouse' + session_id + '_' + phase + '_ext'
    datasets.append((filename, rc_t, rcov, neurons))


    # load CA dataset
    dataset = 'maze15_half-'
    rcov, neurons, tbin, resamples, rc_t, max_count, metainfo, model_name = CA_dataset(dataset, 1)

    filename = 'hc5_' + dataset + '_ext'
    datasets.append((filename, rc_t, rcov, neurons))



    # process
    for data in datasets:
        filename, rc_t, rcov, neurons = data

        sep_t_spike = []
        for k in range(neurons):
            # print((rc_t[k] > 1).sum()) # corrections
            sep_t_spike.append(utils.neural.binned_to_indices((rc_t > 0)[k]))

        # generate past ISIs
        hist_len = 1 # only use instantaneous covariates
        cov = lib.bpp.pp_covariates(rcov, sep_t_spike, max_Horder, hist_len, tbin).numpy()

        # save additional data
        np.savez_compressed('../data/'+filename, covariates=cov)
        
        
if __name__ == "__main__":
    main()
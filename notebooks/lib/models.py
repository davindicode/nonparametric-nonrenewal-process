import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import argparse

import pickle
import os
import sys
sys.path.append("../../neuroppl/")

import neuroppl as nppl
from neuroppl import utils

import helper






### component functions ###
def get_basis(basis_mode='ew'):
    
    if basis_mode == 'id':
        basis = (lambda x: x,)
    
    elif basis_mode == 'ew': # element-wise
        basis = (lambda x: x, lambda x: torch.exp(x))
        
    elif basis_mode == 'qd': # quadratic
        def mix(x):
            N = x.shape[-1]
            out = torch.empty((*x.shape[:-1], N*(N-1)//2), dtype=x.dtype).to(x.device)
            k = 0
            for n in range(1, N):
                for n_ in range(n):
                    out[..., k] = x[..., n]*x[..., n_]
                    k += 1
                
            return out
        
        basis = (lambda x: x, lambda x: x**2, lambda x: torch.exp(x), lambda x: mix(x))
    
    else:
        raise ValueError('Invalid basis expansion')
    
    return basis



class net(nn.Module):
    def __init__(self, C, basis, max_count, channels, shared_W=False):
        super().__init__()
        self.basis = basis
        self.C = C
        expand_C = torch.cat([f_(torch.ones(1, self.C)) for f_ in self.basis], dim=-1).shape[-1]
        
        mnet = nppl.neural_nets.networks.Parallel_MLP(
            [], expand_C, (max_count+1), channels, shared_W=shared_W, 
            nonlin=nppl.neural_nets.networks.Siren(), out=None
        )
        self.add_module('mnet', mnet)
        
        
    def forward(self, input, neuron):
        """
        :param torch.tensor input: input of shape (samplesxtime, channelsxin_dims)
        """
        input = input.view(input.shape[0], -1, self.C)
        input = torch.cat([f_(input) for f_ in self.basis], dim=-1)
        out = self.mnet(input, neuron)
        return out.view(out.shape[0], -1) # t, NxK
    
    
    


    

### model components ###
def temporal_GP(resamples, tbin, batch_info, num_induc, out_dims, tensor_type, ini_var=1., ini_l=1.):
    """
    Temporal GP prior
    """
    out_dims = 1
    in_dims = 1
    Xu = torch.linspace(0, resamples*tbin, num_induc)[None, :, None].repeat(in_dims, 1, 1)
    
    v = ini_var*torch.ones(1, out_dims)
    l = ini_l*torch.ones(out_dims)
    krn1 = kernels.kernel.Constant(variance=v, tensor_type=tensor_type)
    krn2 = kernels.kernel.SquaredExponential(
        input_dims=len(l), lengthscale=l, \
        track_dims=[0], topology='euclid', f='exp', \
        tensor_type=tensor_type
    )
    kernelobj = kernels.kernel.Product(krn1, krn2)
    inducing_points = kernels.kernel.inducing_points(out_dims, Xu, constraints=[])
        
    gp = nppl.mappings.SVGP(
        in_dims, out_dims, kernelobj, inducing_points=inducing_points, 
        whiten=True, jitter=1e-5, mean=torch.zeros(out_dims), learn_mean=True
    )
        
    times_input = torch.arange(resamples)*tbin
    input_group = nppl.inference.input_group()
    input_group.set_XZ([times_input], resamples, batch_info=batch_info)
    
    return nppl.inference.probabilistic_mapping(input_group, gp)

    
    
def coupling_filter(filt_mode, tbin, neurons, hist_len, tensor_type):
    """
    Create the spike coupling object
    """
    if filt_mode[4:8] == 'svgp':
        num_induc = int(filt_mode[8:])
        
        if filt_mode[:4] == 'full':
            D = neurons*neurons
            out_dims = neurons
        elif filt_mode[:4] == 'self':
            D = neurons
            out_dims = 1
            
        v = 100*tbin*torch.ones(D)
        l = 100.*tbin*torch.ones((1, D))
        l_b = 100.*tbin*torch.ones((1, D))
        beta = 1.*torch.ones(D)
        
        mean_func = nppl.mappings.means.decaying_exponential(D, 0., 100.*tbin)

        Xu = torch.linspace(0, hist_len*tbin, num_induc)[None, :, None].repeat(D, 1, 1)
    
        krn1 = nppl.kernels.kernel.Constant(variance=v, tensor_type=tensor_type)
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
        
        hist_couple = nppl.likelihoods.filters.filter_model(out_dims, neurons, hist_len+1, tbin, gp, tens_type=tensor_type)
        
        
    elif filt_mode[4:7] == 'rcb':
        strs = filt_mode[7:].split('-')
        B, L, a, c = int(strs[0]), float(strs[1]), torch.tensor(float(strs[2])), torch.tensor(float(strs[3]))
        
        ini_var = 1.
        if filt_mode[:4] == 'full':
            phi_h = torch.linspace(0., L, B)[:, None, None].repeat(1, neurons, neurons)
            w_h = np.sqrt(ini_var)*torch.randn(B, neurons, neurons)

        elif filt_mode[:4] == 'self':
            phi_h = torch.linspace(0., L, B)[:, None].repeat(1, neurons)
            w_h = np.sqrt(ini_var)*torch.randn(B, neurons)
            
        hist_couple = nppl.likelihoods.filters.raised_cosine_bumps(a=a, c=c, phi=phi_h, w=w_h, timesteps=hist_len+1, 
                                                                   learnable=[False, False, False, True], tensor_type=tensor_type)
        
    else:
        raise ValueError
    
    return hist_couple




def get_likelihood(ll_mode, inner_dims, inv_link, tbin, J, cutoff, mapping_net, C, hgp, tensor_type):
    """
    Create the likelihood object.
    """  
    if ll_mode == 'IBP':
        likelihood = nppl.likelihoods.Bernoulli(tbin, inner_dims, tensor_type=tensor_type)
        
    elif ll_mode == 'IP':
        likelihood = nppl.likelihoods.Poisson(tbin, inner_dims, inv_link, tensor_type=tensor_type)
        
    elif ll_mode == 'ZIP':
        alpha = .1*torch.ones(inner_dims)
        likelihood = nppl.likelihoods.ZI_Poisson(tbin, inner_dims, inv_link, alpha, tensor_type=tensor_type)
        
    elif ll_mode =='hZIP':
        likelihood = nppl.likelihoods.hZI_Poisson(tbin, inner_dims, inv_link, hgp, tensor_type=tensor_type)
        
    elif ll_mode == 'NB':
        r_inv = 10.*torch.ones(inner_dims)
        likelihood = nppl.likelihoods.Negative_binomial(tbin, inner_dims, inv_link, r_inv, tensor_type=tensor_type)
        
    elif ll_mode =='hNB':
        likelihood = nppl.likelihoods.hNegative_binomial(tbin, inner_dims, inv_link, hgp, tensor_type=tensor_type)
        
    elif ll_mode == 'CMP':
        log_nu = torch.zeros(inner_dims)
        likelihood = nppl.likelihoods.COM_Poisson(tbin, inner_dims, inv_link, log_nu, J=J, tensor_type=tensor_type)
        
    elif ll_mode =='hCMP':
        likelihood = nppl.likelihoods.hCOM_Poisson(tbin, inner_dims, inv_link, hgp, J=J, tensor_type=tensor_type)
        
    elif ll_mode == 'IPP':
        likelihood = nppl.likelihoods.Poisson_pp(tbin, inner_dims, inv_link, tensor_type=tensor_type)
        
    elif ll_mode == 'IG':
        shape = torch.ones(inner_dims)
        likelihood = nppl.likelihoods.Gamma(tbin, inner_dims, inv_link, shape, 
                                            allow_duplicate=True, tensor_type=tensor_type)
        
    elif ll_mode == 'IIG':
        mu_t = torch.ones(inner_dims)
        likelihood = nppl.likelihoods.inv_Gaussian(tbin, inner_dims, inv_link, mu_t, 
                                                   allow_duplicate=True, tensor_type=tensor_type)
        
    elif ll_mode == 'LN':
        sigma_t = torch.ones(inner_dims)
        likelihood = nppl.likelihoods.log_Normal(tbin, inner_dims, inv_link, sigma_t, 
                                                 allow_duplicate=True, tensor_type=tensor_type)
        
    elif ll_mode[0] == 'U':
        likelihood = nppl.likelihoods.Universal(inner_dims//C, C, inv_link, cutoff, mapping_net, tensor_type=tensor_type)
        
    else:
        raise NotImplementedError
        
    return likelihood



### model pipeline ###
def inputs_used(x_mode, z_mode, cov_tuple, behav_info, tensor_type):
    """
    Get inputs from model
    """
    raise NotImplementedError # 
    
    
    
def enc_used(map_mode, x_mode, z_mode, cov_tuple, inner_dims, tensor_type):
    """
    Function for generating encoding model
    """
    raise NotImplemetedError
    


def setup_model(data, m, inputs_used, enc_used, tensor_type=torch.float, jitter=1e-5, J=100):
    """"
    Assemble the encoding model
    """
    neurons, tbin, timesamples, max_count, spktrain, cov, batch_info, behav_series = data
    ll_mode, filt_mode, map_mode, x_mode, z_mode, hist_len, folds, delays = m
    
    # checks
    if filt_mode == '' and hist_len > 0:
        raise ValueError
    
    # get settings
    if ll_mode[0] == 'U':
        inv_link = 'identity'
        
        basis_mode = ll_mode[1:3]
        basis = get_basis(basis_mode)
        C = int(ll_mode[3:])
        mapping_net = net(C, basis, max_count, neurons, False)
    
    else:
        if ll_mode == 'IBP':
            inv_link = lambda x: torch.sigmoid(x)/tbin
        elif ll_mode[-3:] == 'exp':
            inv_link = 'exp'
            ll_mode = ll_mode[:-3]
        elif ll_mode[-3:] == 'spl':
            inv_link = 'softplus'
            ll_mode = ll_mode[:-3]
        else:
            raise ValueError('Likelihood inverse link function not defined')
            
        basis_mode = None
        mapping_net = None
        C = 1
        
    # inputs
    inner_dims = neurons*C # number of output dimensions of the input_mapping
    behav_info = (batch_info, neurons, behav_series)
    input_data, d_x, d_z = inputs_used(x_mode, z_mode, cov, behav_info, tensor_type)
    in_dims = d_x + d_z
    
    input_group = nppl.inference.input_group(tensor_type)
    input_group.set_XZ(input_data, timesamples, batch_info=batch_info)
    
    # encoder mapping
    mapping, hgp = enc_used(map_mode, ll_mode, x_mode, z_mode, cov, in_dims, inner_dims, jitter, tensor_type)

    # likelihood
    likelihood = get_likelihood(ll_mode, inner_dims, inv_link, tbin, J, max_count, mapping_net, C, hgp, tensor_type)
    if filt_mode != '':
        filterobj = coupling_filter(filt_mode, tbin, neurons, hist_len, tensor_type)
        likelihood = nppl.likelihoods.filters.filtered_likelihood(likelihood, filterobj)
    likelihood.set_Y(torch.from_numpy(spktrain), batch_info=batch_info)
    
    full = nppl.inference.VI_optimized(input_group, mapping, likelihood)
    return full



### script
def gen_name(mdl_name, m, binsize, max_count, delay, kcv):
    ll_mode, filt_mode, map_mode, x_mode, z_mode, hist_len, folds, delays = m
    delaystr = ''.join(str(d) for d in delays)
    
    name = mdl_name + '{}_{}H{}_{}_X[{}]_Z[{}]_{}K{}_{}d{}_{}f{}'.format(
        ll_mode, filt_mode, hist_len, map_mode, x_mode, z_mode, binsize, max_count, delaystr, delay, folds, kcv, 
    )
    return name



def standard_parser(usage, description):
    """
    Parser arguments belonging to training loop
    """
    parser = argparse.ArgumentParser(
        usage=usage, description=description
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    
    parser.add_argument('--tensor_type', default='float', action='store', type=str)
    
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--cv', nargs='+', type=int)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--bin_size', type=int)
    parser.add_argument('--single_spikes', dest='single_spikes', action='store_true')
    parser.set_defaults(single_spikes=False)
    parser.add_argument('--edge_bins', default=100, type=int)
    
    parser.add_argument('--ncvx', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cov_MC', default=1, type=int)
    parser.add_argument('--ll_MC', default=10, type=int)
    parser.add_argument('--integral_mode', default='MC', action='store', type=str)
    
    parser.add_argument('--jitter', default=1e-5, type=float)
    parser.add_argument('--max_epochs', default=3000, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_2', default=1e-3, type=float)
    
    parser.add_argument('--scheduler_factor', default=0.9, type=float)
    parser.add_argument('--scheduler_interval', default=100, type=int)
    parser.add_argument('--loss_margin', default=-1e0, type=float)
    parser.add_argument('--margin_epochs', default=100, type=int)
    
    parser.add_argument('--likelihood', action='store', type=str)
    parser.add_argument('--filter', default='', action='store', type=str)
    parser.add_argument('--mapping', default='', action='store', type=str)
    parser.add_argument('--x_mode', default='', action='store', type=str)
    parser.add_argument('--z_mode', default='', action='store', type=str)
    parser.add_argument('--delays', nargs='+', default=[0], type=int)
    parser.add_argument('--hist_len', default=0, type=int)
    return parser
    

    
def preprocess_data(folds, delays, cv_runs, batchsize, rc_t, 
                    resamples, rcov, hist_len, has_latent=False, continual=True):
    """
    Returns delay shifted cross-validated data for training
    rcov list of arrays of shape (neurons, time, 1)
    rc_t array of shape (trials, neurons, time) or (neurons, time)
    """
    returns = []
    
    if delays != [0]:
        if min(delays) > 0:
            raise ValueError('Delay minimum must be 0 or less')
        if max(delays) < 0:
            raise ValueError('Delay maximum must be 0 or less')
            
        D_min = -min(delays)
        D_max = -max(delays)
        dd = delays
        
    else:
        D_min = 0
        D_max = 0
        dd = [0]
        
    # history of spike train filter
    rcov = [rc[hist_len:] for rc in rcov]
    resamples -= hist_len
    
    D = -D_max+D_min # total delay steps - 1
    for delay in dd:
        
        # delays
        rc_t_ = rc_t[..., D_min:(D_max if D_max < 0 else None)]
        _min = D_min+delay
        _max = D_max+delay
        
        rcov_ = [rc[_min:(_max if _max < 0 else None)] for rc in rcov]
        resamples_ = resamples - D
        
        cv_sets, vstart = utils.neural.spiketrain_CV(folds, rc_t_, resamples_, rcov_, spk_hist_len=hist_len)
        for kcv in cv_runs:
            if kcv >= 0:
                ftrain, fcov, vtrain, vcov = cv_sets[kcv]

                if continual and has_latent: # has latent and CV, remove validation segment
                    segment_lengths = [vstart[kcv], resamples-vstart[kcv]-vtrain.shape[-1]]
                    trial_ids = [1]*len(segment_lengths)
                    batch_info = utils.neural.batch_segments(segment_lengths, trial_ids, batchsize)
                else:
                    batch_info = batchsize

            else:
                ftrain, fcov = rc_t_, rcov_
                vtrain, vcov = None, None
                batch_info = batchsize

            returns.append((kcv, delay, ftrain, fcov, vtrain, vcov, batch_info))
        
    return returns
    
    

def training(dev, parser, dataset_tuple, inputs_used, enc_used):
    """
    General training loop
    """
    nonconvex_trials = parser.ncvx
    
    if parser.tensor_type == 'float':
        tensor_type = torch.float
    elif parser.tensor_type == 'double':
        tensor_type = torch.double
    else:
        raise ValueError('Invalid tensor type in arguments')

    folds = parser.cv_folds
    delays = parser.delays
    cv_runs = parser.cv
    batchsize = parser.batch_size
    binsize = parser.bin_size
    exclude_edge_bins = parser.edge_bins
    integral_mode = parser.integral_mode
    
    # mode
    ll_mode = parser.likelihood
    filt_mode = parser.filter
    map_mode = parser.mapping
    x_mode = parser.x_mode
    z_mode = parser.z_mode
    hist_len = parser.hist_len
    
    m = (ll_mode, filt_mode, map_mode, x_mode, z_mode, hist_len, folds, delays)
    rcov_orig, units_used, tbin, resamples, spktrain, max_count, bin_size, metainfo, mdl_name = dataset_tuple
    
    # removes edges for warping
    if exclude_edge_bins > 0:
        spktrain = spktrain[..., exclude_edge_bins:-exclude_edge_bins]
        rcov = tuple(rc[exclude_edge_bins:-exclude_edge_bins] for rc in rcov_orig) # covariates are one dimensional arrays
        resamples = spktrain.shape[-1]
    else:
        rcov = rcov_orig

    ### training
    has_latent = False if z_mode == '' else True
    for cvdata in preprocess_data(folds, delays, cv_runs, batchsize, spktrain, resamples, rcov, hist_len, has_latent):
        kcv, delay, ftrain, fcov, vtrain, vcov, batch_info = cvdata
        data = (units_used, tbin, fcov[0].shape[0], max_count, ftrain, fcov, batch_info, rcov_orig)
        model_name = gen_name(mdl_name, m, bin_size, max_count, delay, kcv)
        print(model_name)
        
        ### fitting
        for kk in range(nonconvex_trials):

            retries = 0
            while True:
                try:
                    # model
                    full_model = setup_model(
                        data, m, inputs_used, enc_used, 
                        tensor_type, jitter=parser.jitter, J=100
                    )
                    full_model.to(dev)

                    # fit
                    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: parser.scheduler_factor)
                    opt_tuple = (optim.Adam, parser.scheduler_interval, sch)
                    opt_lr_dict = {'default': parser.lr}
                    if z_mode == 'T1':
                        opt_lr_dict['mapping.kernel.kern1._lengthscale'] = parser.lr_2
                    for z_dim in full_model.input_group.latent_dims:
                        opt_lr_dict['input_{}.finv_std'.format(z_dim)] = parser.lr_2

                    full_model.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

                    annealing = lambda x: 1.0
                    losses = full_model.fit(parser.max_epochs, loss_margin=parser.loss_margin, 
                                            margin_epochs=parser.margin_epochs, kl_anneal_func=annealing, 
                                            cov_samples=parser.cov_MC, ll_samples=parser.ll_MC, ll_mode=integral_mode)
                    break
                    
                except RuntimeError as e:
                    print(e)
                    print('Retrying...')
                    if retries == 3: # max retries
                        print('Stopped after max retries.')
                        sys.exit()
                    retries += 1

            ### save and progress
            if os.path.exists('./checkpoint/best_fits'): # check previous best losses
                with open('./checkpoint/best_fits', 'rb') as f:
                    best_fits = pickle.load(f)
                    if model_name in best_fits:
                        lowest_loss = best_fits[model_name]
                    else:
                        lowest_loss = np.inf # nonconvex pick the best
            else:
                best_fits = {}          
                lowest_loss = np.inf # nonconvex pick the best

            if losses[-1] < lowest_loss:
                # save model
                if not os.path.exists('./checkpoint'):
                    os.makedirs('./checkpoint')
                torch.save({'full_model': full_model.state_dict()}, './checkpoint/' + model_name)
                
                best_fits[model_name] = losses[-1] # add new fit
                with open('./checkpoint/best_fits', 'wb') as f:
                    pickle.dump(best_fits, f)

                
                
def load_model(checkpoint_dir, m, dataset_tuple, inputs_used, enc_used, 
               delay, cv_run, batch_info, gpu, tensor_type=torch.float, jitter=1e-5, J=100):
    """
    Load the model with cross-validated data structure
    """
    ll_mode, filt_mode, map_mode, x_mode, z_mode, hist_len, folds, delays = m
    rcov_orig, units_used, tbin, resamples, spktrain, max_count, bin_size, metainfo, mdl_name = dataset_tuple
    
    # removes edges for warping
    if exclude_edge_bins > 0:
        spktrain = spktrain[..., exclude_edge_bins:-exclude_edge_bins]
        rcov = tuple(rc[exclude_edge_bins:-exclude_edge_bins] for rc in rcov_orig) # covariates are one dimensional arrays
        resamples = spktrain.shape[-1]
    else:
        rcov = rcov_orig
    
    ### training loop
    has_latent = False if z_mode == '' else True
    cvdata = preprocess_data(folds, [delay], [cv_run], batch_info, spktrain, resamples, rcov, hist_len, has_latent)[0]
    
    _, _, ftrain, fcov, vtrain, vcov, cvbatch_info = cvdata
    fit_data = (units_used, tbin, fcov[0].shape[0], max_count, ftrain, fcov, cvbatch_info)
    
    fcov_ = inputs_used(x_mode, z_mode, fcov, batch_info, tensor_type)[0]
    fit_set = (fcov_, torch.from_numpy(ftrain))
    vcov_ = inputs_used(x_mode, z_mode, vcov, batch_info, tensor_type)[0] if vcov is not None else None
    validation_set = (vcov_, torch.from_numpy(vtrain)) if vtrain is not None else None
    
    ### model
    full_model = setup_model(fit_data, m, inputs_used, enc_used, 
                             jitter=jitter, J=J)
    full_model.to(gpu)

    ### load
    model_name = gen_name(mdl_name, m, bin_size, max_count, delay, cv_run)
    checkpoint = torch.load(checkpoint_dir + model_name, map_location='cuda:{}'.format(gpu))
    full_model.load_state_dict(checkpoint['full_model'])
    return full_model, fit_set, validation_set
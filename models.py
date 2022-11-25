import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
sys.path.append("../..")

import neuroprob.models as mdl
from neuroprob import neural_utils, stats
import neuroprob.models.RNN as rt





# setup GP parameters
def prior_GP(mode, resamples, tbin, bs):
    if mode != 'hdxR1gp' and mode != 'T1gp':
        return None
    
    if mode == 'T1gp':
        resamples -= 1
    
    units_ = 1
    num_induc = 64
    inducing_points = np.array([np.linspace(0, resamples*tbin, num_induc)]).T[None, ...].repeat(units_, axis=0)

    l = 100*tbin*np.ones(units_)
    v = np.ones(units_)
    kernels_tuples = [('OU', 'euclid', np.array([l]))]
    gp_base = mdl.nonparametrics.Gaussian_process(units_, inducing_points, kernels_tuples, 
                                            [(None, None, None, units_)], # mean fixed to zero by default
                                            shared_kernel_params=False, cov_type='factorized',
                                            inv_link='identity', whiten=True)

    gp_base.set_params(tbin, jitter=1e-5)

    times_input = np.arange(resamples)*tbin
    gp_base.preprocess([times_input], resamples, batch_size=bs)
    return gp_base



def cov_used(mode, behav_tuple, GP_prior=None):
    """
    Create the used covariates list.
    """
    x_t, y_t, th_t, s_t, hd_t, w_t, time_t = behav_tuple
    resamples = x_t.shape[0]
    
    if mode == 'pos':
        covariates = [x_t, y_t]
        
    elif mode == 'pos_th':
        covariates = [x_t, y_t, th_t]
        
    elif mode == 'pos_th_s':
        covariates = [x_t, y_t, th_t, s_t]
        
    elif mode == 'pos_th_s_hd':
        covariates = [x_t, y_t, th_t, s_t, hd_t]
        
    elif mode is None:
        covariates = []
        
    else:
        raise ValueError
        
    return covariates



def kernel_used(mode, behav_tuple, outdims, var=1.):
    x_t, y_t, th_t, s_t, hd_t, w_t, time_t = behav_tuple
    
    l_ang = 5.*np.ones(outdims)
    l_w = w_t.std()*np.ones(outdims)
    l_s = 10.*np.ones(outdims)
    l = 10.*np.ones(outdims)
    v = var*np.ones(outdims)
    l_time = time_t.max()/2.*np.ones(outdims)
    
    factorized = 1
    if mode == 'pos':
        kernel_tuples = [
            ('variance', v), 
            ('RBF', 'euclid', np.array([l, l]))
        ]
    
    elif mode == 'pos_th':
        kernel_tuples = [
            ('variance', v), 
            ('RBF', 'euclid', np.array([l, l])), 
            ('RBF', 'torus', np.array([l_ang]))
        ]
        
    elif mode == 'pos_th_s':
        kernel_tuples = [
            ('variance', v), 
            ('RBF', 'euclid', np.array([l, l])), 
            ('RBF', 'torus', np.array([l_ang])), 
            ('RBF', 'euclid', np.array([l_s]))
        ]
        
    elif mode == 'pos_th_s_hd':
        kernel_tuples = [
            ('variance', v), 
            ('RBF', 'euclid', np.array([l, l])), 
            ('RBF', 'torus', np.array([l_ang])), 
            ('RBF', 'euclid', np.array([l_s])), 
            ('RBF', 'torus', np.array([l_ang]))
        ]
        
    elif mode is None:
        v = np.ones(outdims)
        kernel_tuples = [('variance', v)]

    else:
        raise ValueError
        
    if factorized == 1:
        kernel_tuples_ = (kernel_tuples,)
    elif factorized == 2:
        kernel_tuples_ = (kt, kt_, kernel_tuples)
    elif factorized == 3:
        kernel_tuples_ = (kt, kt_, kt__, kernel_tuples)
        
    return kernel_tuples_



def GP_params(mode, behav_tuple, num_induc, neurons, inv_link, tbin, jitter, mean_func, filter_data, 
              cov_type='factorized', shared_kernel_params=False, var=1., GP_prior=None):
    """
    Create the GP object.
    """
    x_t, y_t, th_t, s_t, hd_t, w_t, time_t = behav_tuple
    resamples = x_t.shape[0]
    covariates = cov_used(mode, behav_tuple, GP_prior)
    kernel_tuples_ = kernel_used(mode, behav_tuple, neurons, var)
    
    left_x = x_t.min()
    right_x = x_t.max()
    bottom_y = y_t.min()
    top_y = y_t.max()
    
    factorized = 1
    if mode == 'pos':
        ind_list = [np.linspace(left_x, right_x, num_induc), 
                    np.linspace(bottom_y, top_y, num_induc)]
        VI_tuples = [(None, None, None, 1)]*len(covariates)
        
    elif mode == 'pos_th':
        ind_list = [np.linspace(left_x, right_x, num_induc), 
                    np.linspace(bottom_y, top_y, num_induc), 
                    np.linspace(0, 2*np.pi, num_induc+1)[:-1]]
        VI_tuples = [(None, None, None, 1)]*len(covariates)
        
    elif mode == 'pos_th_s':
        ind_list = [np.linspace(left_x, right_x, num_induc), 
                    np.linspace(bottom_y, top_y, num_induc), 
                    np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.linspace(0, s_t.std(), num_induc)]
        VI_tuples = [(None, None, None, 1)]*len(covariates)
        
    elif mode == 'pos_th_s_hd':
        ind_list = [np.linspace(left_x, right_x, num_induc), 
                    np.linspace(bottom_y, top_y, num_induc), 
                    np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.linspace(0, s_t.std(), num_induc), 
                    np.linspace(0, 2*np.pi, num_induc+1)[:-1]]
        VI_tuples = [(None, None, None, 1)]*len(covariates)

    elif mode is None:
        VI_tuples = []
        ind_list = []

    else:
        raise ValueError
        
    if filter_data is not None: # add filter time dimension
        filter_len, filter_kernel = filter_data
        VI_tuples = [(None, None, None, 1)] + VI_tuples
        max_time = filter_len*tbin
        ind_list = [np.linspace(0, max_time, num_induc)] + ind_list
        temp = []
        for k in kernel_tuples_:
            temp.append(k[:1] + filter_kernel + k[1:])
        kernel_tuples_ = temp
        
    if factorized == 2:
        kt, kt_, kernel_tuples = kernel_tuples_
        inducing_points = np.array(ind_list).T[None, ...].repeat(neurons, axis=0)
        rate_model = mdl.nonparametrics.Gaussian_process(neurons, inducing_points, kt,
                                                    VI_tuples, inv_link=inv_link, shared_kernel_params=shared_kernel_params,
                                                    cov_type=cov_type, whiten=True, 
                                                    mean=mean_func)
        
        rate_model_ = mdl.nonparametrics.Gaussian_process(neurons, inducing_points, kt_,
                                                    VI_tuples, inv_link=inv_link, shared_kernel_params=shared_kernel_params,
                                                    cov_type=cov_type, whiten=True, 
                                                    mean=mean_func)
        
        rate_model.set_params(tbin, jitter=jitter)
        rate_model_.set_params(tbin, jitter=jitter)
        glm_rate = mdl.parametrics.product_model([rate_model, rate_model_])
        
    elif factorized == 3:
        kt, kt_, kt__, kernel_tuples = kernel_tuples_
        inducing_points = np.array(ind_list).T[None, ...].repeat(neurons, axis=0)
        rate_model = mdl.nonparametrics.Gaussian_process(neurons, inducing_points, kt,
                                                    VI_tuples, inv_link=inv_link, shared_kernel_params=shared_kernel_params,
                                                    cov_type=cov_type, whiten=True, 
                                                    mean=mean_func)
        
        rate_model_ = mdl.nonparametrics.Gaussian_process(neurons, inducing_points, kt_,
                                                    VI_tuples, inv_link=inv_link, shared_kernel_params=shared_kernel_params,
                                                    cov_type=cov_type, whiten=True, 
                                                    mean=mean_func)
        
        rate_model__ = mdl.nonparametrics.Gaussian_process(neurons, inducing_points, kt__,
                                                    VI_tuples, inv_link=inv_link, shared_kernel_params=shared_kernel_params,
                                                    cov_type=cov_type, whiten=True, 
                                                    mean=mean_func)
        
        rate_model.set_params(tbin, jitter=jitter)
        rate_model_.set_params(tbin, jitter=jitter)
        rate_model__.set_params(tbin, jitter=jitter)
        glm_rate = mdl.parametrics.product_model([rate_model, rate_model_, rate_model__])
        
    else:
        kernel_tuples, = kernel_tuples_
        inducing_points = np.array(ind_list).T[None, ...].repeat(neurons, axis=0)
        glm_rate = mdl.nonparametrics.Gaussian_process(neurons, inducing_points, kernel_tuples,
                                                    VI_tuples, inv_link=inv_link, shared_kernel_params=shared_kernel_params,
                                                    cov_type=cov_type, whiten=True, 
                                                    mean=mean_func)
        glm_rate.set_params(tbin, jitter=jitter)
        
    return glm_rate



def filter_params(spkcoupling_mode, mode, num_induc, behav_tuple, tbin, neurons, hist_len, filter_props, jitter):
    """
    Create the spike coupling object.
    """
    if spkcoupling_mode is None:
        return None
    
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
        mean_func = mdl.filters.decaying_exponential(D, 0., l_mean0)
        filter_data = (hist_len, filter_kernel)
        
    elif spkcoupling_mode[:3] == 'rcb':
        a, c, B, L, ini_var = filter_props
        if spkcoupling_mode[3:7] == 'full':
            phi_h = np.linspace(0., L, B)[:, None, None].repeat(neurons, axis=1).repeat(neurons, axis=2)
        else:
            phi_h = np.linspace(0., L, B)[:, None].repeat(neurons, axis=1)
        
    else:
        raise ValueError
    
    if spkcoupling_mode == 'rcbself':
        w_h = np.sqrt(ini_var)*np.random.randn(B, neurons)
        hist_couple = mdl.filters.raised_cosine_bumps(a=a, c=c, phi=phi_h, w=w_h, timesteps=hist_len, 
                                                      learnable=[False, False, False, True])
    elif spkcoupling_mode == 'rcbfull':
        w_h = np.sqrt(ini_var)*np.random.randn(B, neurons, neurons)
        hist_couple = mdl.filters.raised_cosine_bumps(a=a, c=c, phi=phi_h, w=w_h, timesteps=hist_len, 
                                                      learnable=[False, False, False, True])
    elif spkcoupling_mode == 'rcbselfh' or spkcoupling_mode == 'rcbselfhc':
        mean_func = np.zeros(B*neurons)
        fm = GP_params(mode, behav_tuple, num_induc, B*neurons, 'identity', tbin, jitter, mean_func, 
                       None, cov_type=None, shared_kernel_params=(spkcoupling_mode[-1]=='c'), var=ini_var)
        
        hist_couple = mdl.filters.hetero_raised_cosine_bumps(a=a, c=c, phi=phi_h, timesteps=hist_len, 
                                                             inner_loop_bs=1000, hetero_model=fm, 
                                                             learnable=[False, False, False])
    elif spkcoupling_mode == 'rcbfullh' or spkcoupling_mode == 'rcbfullhc':
        mean_func = np.zeros(B*neurons*neurons)
        fm = GP_params(mode, behav_tuple, num_induc, B*neurons*neurons, 'identity', tbin, jitter, mean_func, 
                       None, cov_type=None, shared_kernel_params=(spkcoupling_mode[-1]=='c'), var=ini_var)
        
        hist_couple = mdl.filters.hetero_raised_cosine_bumps(a=a, c=c, phi=phi_h, timesteps=hist_len, 
                                                             hetero_model=fm, inner_loop_bs=100, 
                                                             learnable=[False, False, False])
    elif spkcoupling_mode == 'gpself':
        filter_data = (hist_len, filter_kernel)
        fm = GP_params(None, behav_tuple, num_induc, neurons, 'identity', tbin, jitter, mean_func, 
                       filter_data)
        
        hist_couple = mdl.filters.filter_model(1, neurons, hist_len, tbin, fm, tens_type=torch.float)
    elif spkcoupling_mode == 'gpselfh' or spkcoupling_mode == 'gpselfhc':
        fm = GP_params(mode, behav_tuple, num_induc, neurons, 'identity', tbin, jitter, mean_func, 
                       filter_data, cov_type=None, shared_kernel_params=(spkcoupling_mode[-1]=='c'))
        
        hist_couple = mdl.filters.hetero_filter_model(1, neurons, hist_len, tbin, fm, 
                                                      inner_loop_bs=100, tens_type=torch.float)
    elif spkcoupling_mode == 'gpfull':
        fm = GP_params(None, behav_tuple, num_induc, neurons*neurons, 'identity', tbin, jitter, mean_func, 
                       filter_data)
        
        hist_couple = mdl.filters.filter_model(neurons, neurons, hist_len, tbin, fm, tens_type=torch.float)
    elif spkcoupling_mode == 'gpfullh' or spkcoupling_mode == 'gpfullhc':
        fm = GP_params(mode, behav_tuple, num_induc, neurons*neurons, 'identity', tbin, jitter, mean_func, 
                       filter_data, cov_type=None, shared_kernel_params=(spkcoupling_mode[-1]=='c'))
        
        hist_couple = mdl.filters.hetero_filter_model(neurons, neurons, hist_len, tbin, fm, 
                                                   inner_loop_bs=100, tens_type=torch.float)
    else:
        raise NotImplementedError

    return hist_couple



def likelihood_params(ll_mode, mode, behav_tuple, num_induc, neurons, inv_link, tbin, jitter, J):
    """
    Create the likelihood object.
    """
    if mode is not None:
        kernel_tuples_ = kernel_used(mode, behav_tuple, neurons)
        factorized = len(kernel_tuples_)
        if factorized > 1: # overwrite
            inv_link = 'relu'
    
    inv_link_hetero = None
    if ll_mode == 'IBP':
        likelihood = mdl.likelihoods.Bernoulli(neurons, inv_link)
    elif ll_mode == 'IP':
        likelihood = mdl.likelihoods.Poisson(neurons, inv_link)
    elif ll_mode == 'ZIP' or ll_mode =='ZIPh':
        alpha = .1*np.ones(neurons)
        likelihood = mdl.likelihoods.ZI_Poisson(neurons, inv_link, alpha)
        if ll_mode =='ZIPh':
            inv_link_hetero = 'sigmoid'
            #inv_link_hetero = lambda x: torch.sigmoid(x)/tbin
    elif ll_mode == 'NB' or ll_mode =='NBh':
        r_inv = 10.*np.ones(neurons)
        likelihood = mdl.likelihoods.Negative_binomial(neurons, inv_link, r_inv)
        if ll_mode =='NBh':
            inv_link_hetero = 'softplus'
    elif ll_mode == 'CMP' or ll_mode =='CMPh':
        log_nu = np.zeros(neurons)
        likelihood = mdl.likelihoods.COM_Poisson(neurons, inv_link, log_nu, J=J)
        if ll_mode =='CMPh':
            inv_link_hetero = 'identity'
            #inv_link_hetero = 'softplus'
    elif ll_mode == 'IG':
        shape = np.ones(neurons)
        likelihood = mdl.likelihoods.Gamma(neurons, inv_link, shape, allow_duplicate=False)
    elif ll_mode == 'IIG':
        mu_t = np.ones(neurons)
        likelihood = mdl.likelihoods.invGaussian(neurons, inv_link, mu_t, allow_duplicate=False)
    elif ll_mode == 'LN':
        sigma_t = np.ones(neurons)
        likelihood = mdl.likelihoods.logNormal(neurons, inv_link, sigma_t, allow_duplicate=False)
    else:
        raise NotImplementedError
        
    if inv_link_hetero is not None:
        mean_func = np.zeros((neurons))
        gp_lvms = GP_params(mode, behav_tuple, num_induc, neurons, inv_link_hetero, tbin, jitter, mean_func, None)
    else:
        gp_lvms = None
        
    likelihood.set_params(tbin)
    
    return likelihood, gp_lvms



def set_glm(mode, ll_mode, spkcoupling_mode, behav_tuple, neurons, tbin, rc_t, num_induc, inv_link='exp', jitter=1e-4,
            batch_size=10000, hist_len=19, filter_props=None, J=100):
    """
    Assemble the GLM object.
    """
    resamples = rc_t.shape[-1]
    if ll_mode == 'IBP': # overwrite
        inv_link = lambda x: torch.sigmoid(x)/tbin
    
    gpp = prior_GP(mode, resamples, tbin, batch_size)
    glm_rate = GP_params(mode, behav_tuple, num_induc, neurons, inv_link, tbin, jitter, np.zeros((neurons)), None, GP_prior=gpp)
    hist_couple = filter_params(spkcoupling_mode, mode, num_induc, behav_tuple, tbin, neurons, hist_len, filter_props, jitter)
    likelihood, gp_lvms = likelihood_params(ll_mode, mode, behav_tuple, num_induc, neurons, inv_link, tbin, jitter, J)
    
    glm = mdl.inference.nll_optimized([glm_rate], likelihood, spk_couple=hist_couple, dispersion_model=gp_lvms)
    covariates = cov_used(mode, behav_tuple, GP_prior=gpp)
    glm.preprocess(covariates, resamples, rc_t, batch_size=batch_size)
    return glm, covariates



# tools
def compute_count_stats(glm, ll_mode, tbin, spktrain, behav_list, neurons, traj_len=None, traj_spikes=None,
                        start=0, T=100000, bs=5000, mode='single'):
    """
    Compute the dispersion statistics, per neuron in a population.
    
    :param string mode: *single* mode refers to computing separate single neuron quantities, *population* mode 
                        refers to computing over a population indicated by neurons, *peer* mode involves the 
                        peer predictability i.e. conditioning on all other neurons given a subset
    """
    assert mode == 'single' or mode == 'peer' or mode == 'population'
    N = int(np.ceil(T/bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start:start+T]
    behav_list = [b[start:start+T] for b in behav_list]

    for k in range(N):
        covariates_ = [b[k*bs:(k+1)*bs] for b in behav_list]
        
        if glm.filter_len > 1:
            ini_train = spktrain[None, :, :glm.filter_len-1]
        else:
            ini_train = np.zeros((1, glm.neurons, 1)) # used only for trial count
        
        if mode == 'single' or mode == 'population':
            ospktrain = spktrain[None, :, glm.filter_len-1:]
        elif mode == 'peer':
            ospktrain = spktrain[None, :, glm.filter_len-1:]
            for ne in neurons:
                ospktrain[ne, :] = 0

        _, rate, disp, _ = glm.sample(covariates_, ini_train, neuron=[], obs_spktrn=ospktrain, 
                                        MC_samples=1000)
        rate_model += [rate[0, neurons, :]]
        if glm.dispersion_model is not None:
            shape_model += [disp[0, neurons, :]]
                

    rate_model = np.concatenate(rate_model, axis=1)
    if glm.dispersion_model is not None:
        shape_model = np.concatenate(shape_model, axis=1)
    
    if ll_mode == 'IP':
        shape_model = None
        f_p = lambda c, avg, shape, t: stats.poiss_count_prob(c, avg, shape, t)
    elif ll_mode[:2] == 'NB':
        if glm.dispersion_model is None:
            shape_model = glm.likelihood.r_inv.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: stats.nb_count_prob(c, avg, shape, t)
    elif ll_mode[:3] == 'CMP':
        if glm.dispersion_model is None:
            shape_model = glm.likelihood.nu.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: stats.cmp_count_prob(c, avg, shape, t)
    elif ll_mode[:3] == 'ZIP':
        if glm.dispersion_model is None:
            shape_model = glm.likelihood.alpha.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: stats.zip_count_prob(c, avg, shape, t)
    else:
        raise ValueError
    m_f = lambda x: x

    if shape_model is not None:
        assert traj_len == 1
    if traj_len is not None:
        traj_lens = (T // traj_len) * [traj_len]
        
    q_ = []
    for k, ne in enumerate(neurons):
        if traj_spikes is not None:
            avg_spikecnt = np.cumsum(rate_model[k]*tbin)
            nc = 1
            traj_len = 0
            for tt in range(T):
                if avg_spikecnt >= traj_spikes*nc:
                    nc += 1
                    traj_lens.append(traj_len)
                    traj_len = 0
                    continue
                traj_len += 1
                
        if shape_model is not None:
            sh = shape_model[k]
            spktr = spktrain[ne]
            rm = rate_model[k]
        else:
            sh = None
            spktr = []
            rm = []
            start = np.cumsum(traj_lens)
            for tt, traj_len in enumerate(traj_lens):
                spktr.append(spktrain[ne][start[tt]:start[tt]+traj_len].sum())
                rm.append(rate_model[k][start[tt]:start[tt]+traj_len].sum())
            spktr = np.array(spktr)
            rm = np.array(rm)
                    
        q_.append(stats.count_KS_method(f_p, m_f, tbin, spktr, rm, shape=sh))
        
    if mode == 'single':
        cnt_tuples = [(q, *stats.KS_statistics(q)) for q in q_]
        I = (rate_model*np.log(rate_model/rate_model.mean(-1, keepdims=True))).mean(-1)
    elif mode == 'population':
        q_tot = np.concatenate(q_)
        cnt_tuples = (q_tot, *stats.KS_statistics(q_tot))
        I = (rate_model*np.log(rate_model/rate_model.mean(-1, keepdims=True))).mean(-1).sum()
    elif mode == 'peer':
        pass
    
    return cnt_tuples, I



def compute_isi_stats(glm, ll_mode, tbin, spktrain, behav_list, neurons, start=0, T=100000, bs=5000):
    """
    Compute the dispersion statistics, per neuron in a population.
    """
    N = int(np.ceil(T/bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start:start+T]
    behav_list = [b[start:start+T] for b in behav_list]

    for k in range(N):
        covariates_ = [b[k*bs:(k+1)*bs] for b in behav_list]
        rate_model += [glm.rate_model[0].eval_rate(covariates_, neurons)]

    rate_model = np.concatenate(rate_model, axis=1)
    
    isi_tuples = []
    
    if ll_mode == 'IP':
        dist_isi = mdl.point_process.ISI_gamma(1.0)
        
    for ne in neurons:
        if ll_mode is not 'IP':
            dist_isi = glm.likelihood.ISI_dist(ne)
        t_spike = neural_utils.BinToTrain(spktrain[ne])
        q = stats.ISI_KS_method(dist_isi, tbin, t_spike, rate_model)
        isi_tuples.append((q, *stats.KS_statistics(q)))

    return isi_tuples



def compute_invisi_stats(glm, ll_mode, tbin, spktrain, behav_list, neurons, start=0, T=100000, bs=5000):
    """
    Compute the dispersion statistics, per neuron in a population.
    """
    N = int(np.ceil(T/bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start:start+T]
    behav_list = [b[start:start+T] for b in behav_list]

    for k in range(N):
        covariates_ = [b[k*bs:(k+1)*bs] for b in behav_list]
        rate_model += [glm.rate_model[0].eval_rate(covariates_, neurons)]

    rate_model = np.concatenate(rate_model, axis=1)
    
    invisi_tuples = []
    
    if ll_mode == 'IP':
        dist_invisi = mdl.point_process.ISI_invgamma(1.0)
        
    for ne in neurons:
        if ll_mode == 'IG':
            shape = glm.likelihood.shape[ne].data.cpu().numpy()
            dist_invisi = mdl.point_process.ISI_invgamma(shape, scale=1./shape)
        t_spike = neural_utils.BinToTrain(spktrain[ne])
        q = stats.invISI_KS_method(dist_invisi, tbin, t_spike, rate_model)
        invisi_tuples.append((q, *stats.KS_statistics(q)))

    return invisi_tuples




def pred_ll(glm, vtrain, vcov, time_steps, neuron=None, ll_mode='GH', ll_samples=100):
    """
    Compute the predictive log likelihood (ELBO).
    """
    glm.preprocess(vcov, time_steps, vtrain, batch_size=time_steps)
    if neuron is None:
        return -glm.nll(0, cov_samples=1, ll_mode=ll_mode, ll_samples=ll_samples).data.cpu().numpy()    
    else:
        cv_ll = []
        for n in neuron:
            cv_ll.append(-glm.nll(0, cov_samples=1, ll_mode=ll_mode, 
                                  ll_samples=ll_samples, neuron=[n]).data.cpu().numpy())
        return np.array(cv_ll)
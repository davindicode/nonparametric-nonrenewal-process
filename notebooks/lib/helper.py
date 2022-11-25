import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
sys.path.append("../../neuroppl/")

import neuroppl as nppl
from neuroppl import utils
from neuroppl import kernels



### model ###
def sample_F(mapping, likelihood, covariates, MC, F_dims, trials=1):
    """
    Sample F from diagonalized variational posterior.
    
    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)

    with torch.no_grad():
        if mapping.MC_only:
            samples = mapping.sample_F(cov)[:, F_dims, :] # TODO: cov_samples vs ll_samples?
            h = samples.view(-1, trials, *samples.shape[1:])
        else:
            F_mu, F_var = mapping.compute_F(cov)
            h = likelihood.mc_gen(F_mu, F_var, MC, F_dims)

    return h



def posterior_rate(mapping, likelihood, covariates, MC, F_dims, trials=1, percentiles=[0.05, .5, 0.95]):
    """
    Sample F from diagonalized variational posterior.
    
    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)
    with torch.no_grad():
        if mapping.MC_only:
            F = mapping.sample_F(cov)[:, F_dims, :] # TODO: cov_samples vs ll_samples?
            samples = likelihood.f(F.view(-1, trials, *samples.shape[1:]))
        else:
            F_mu, F_var = mapping.compute_F(cov)
            samples = likelihood.sample_rate(
                F_mu[:, F_dims, :], F_var[:, F_dims, :], trials, MC)
    
    return utils.signal.percentiles_from_samples(samples, percentiles)
    
    
    
def sample_tuning_curves(mapping, likelihood, covariates, MC, F_dims, trials=1):
    """
    """
    cov = mapping.to_XZ(covariates, trials)
    with torch.no_grad():
        eps = torch.randn((MC*trials, *cov.shape[1:-1]), 
                          dtype=mapping.tensor_type, device=mapping.dummy.device)
        mapping.jitter = 1e-4
        samples = mapping.sample_F(cov, eps)
        T = samples.view(-1, trials, *samples.shape[1:])

    return T



def sample_Y(mapping, likelihood, covariates, trials, MC=1):
    """
    Sampling gives np.ndarray
    """
    cov = mapping.to_XZ(covariates, trials)
    
    with torch.no_grad():
            
        F_mu, F_var = mapping.compute_F(cov)
        rate = likelihood.sample_rate(F_mu, F_var, trials, MC) # MC, trials, neuron, time

        rate = rate.mean(0).cpu().numpy()
        syn_train = likelihood.sample(rate, XZ=cov)
        
    return syn_train



### GP
class time_transform(nn.Module):
    
    def __init__(self, tau_0):
        super().__init__()
        self.register_parameter('tau_0', Parameter(tau_0))
        
    def forward(self, t):
        """
        Input of shape (mc, neurons, time, dims)
        """
        tau_0 = self.tau_0[None, :, None, :]
        a = torch.log(t/tau_0+1.)
        return torch.log(t/tau_0+1.)
    
    
    
def latent_kernel(z_mode):
    """
    """
    ind_list = []
    kernel_tuples = []
    
    if z_mode[:1] == 'R':
        dz = int(z_mode[1:]) 
        for h in range(dz):
            ind_list += [np.random.randn(num_induc)]
        kernel_tuples += [('SE', 'euclid', torch.tensor([l_one]*dz))]
        
    elif z_mode != '':
        raise ValueError
        
    return kernel_tuples, ind_list
    
    

def create_kernel(kernel_tuples, kern_f, tensor_type):
    """
    Helper function for creating kernel triplet tuple
    """
    track_dims = 0
    kernelobj = 0

    constraints = []
    for k, k_tuple in enumerate(kernel_tuples):

        if k_tuple[0] is not None:

            if k_tuple[0] == 'variance':
                krn = kernels.kernel.Constant(variance=k_tuple[1], tensor_type=tensor_type)

            else:
                kernel_type = k_tuple[0]
                topology = k_tuple[1]
                lengthscales = k_tuple[2]

                if topology == 'sphere':
                    constraints += [(track_dims, track_dims+len(lengthscales), 'sphere'),]

                act = []
                for _ in lengthscales:
                    act += [track_dims]
                    track_dims += 1

                if kernel_type == 'SE':
                    krn = kernels.kernel.SquaredExponential(
                        input_dims=len(lengthscales), lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'tSE':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    tau_0 = k_tuple[3]
                    kern = kernels.kernel.SquaredExponential(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        topology=topology, f=kern_f, tensor_type=tensor_type
                    )
                    mapping = time_transform(torch.tensor(tau_0, dtype=tensor_type))
                    krn = kernels.kernel.DeepKernel(len(lengthscales), kern, mapping, track_dims=act)
                    
                elif kernel_type == 'DSE':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    lengthscale_beta = k_tuple[3]
                    beta = k_tuple[4]
                    krn = kernels.kernel.DSE(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        lengthscale_beta=lengthscale_beta, \
                        beta=beta, \
                        track_dims=act, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'OU':
                    krn = kernels.kernel.Exponential(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'RQ':
                    scale_mixture = k_tuple[3]
                    krn = kernels.kernel.RationalQuadratic(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        scale_mixture=scale_mixture, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'Matern32':
                    krn = kernels.kernel.Matern32(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'Matern52':
                    krn = kernels.kernel.Matern52(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'linear':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    krn = kernels.kernel.Linear(
                        input_dims=len(lengthscales), \
                        track_dims=act, f=kern_f
                    )
                    
                elif kernel_type == 'polynomial':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    degree = k_tuple[3]
                    krn = kernels.kernel.Polynomial(
                        input_dims=len(lengthscales), \
                        bias=lengthscales, \
                        degree=degree, track_dims=act, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                else:
                    raise NotImplementedError('Kernel type is not supported.')

            kernelobj = kernels.kernel.Product(kernelobj, krn) if kernelobj != 0 else krn

        else:
            track_dims += 1

    return kernelobj, constraints




### latent
def latent_objects(z_mode, d_x, timesamples, tensor_type):
    """
    """ 
    if z_mode[:1] == 'R':
        d_z = int(z_mode[1:])
        
        if d_z == 1:
            p = nppl.inputs.priors.Autoregressive(
                torch.tensor(0.), torch.tensor(4.), 'euclid', 1, p=1, tensor_type=tensor_type)
        else:
            p = nppl.inputs.priors.Autoregressive(
                torch.tensor([0.]*d_z), torch.tensor([4.]*d_z), 'euclid', d_z, p=1, tensor_type=tensor_type)
            
        v = nppl.inputs.variational.IndNormal(
            torch.rand(timesamples, d_z)*0.1, torch.ones((timesamples, d_z))*0.01, 'euclid', d_z, tensor_type=tensor_type)
        
        latents = [nppl.inference.prior_variational_pair(d_z, p, v)]
        
    elif z_mode[:1] == 'T':
        d_z = int(z_mode[1:])
        
        if d_z == 1:
            p = nppl.inputs.priors.Autoregressive(
                torch.tensor(0.), torch.tensor(4.0), 'torus', 1, p=1, tensor_type=tensor_type)
        else:
            p = nppl.inputs.priors.Autoregressive(
                torch.tensor([0.]*d_z), torch.tensor([4.]*d_z), 'euclid', d_z, p=1, tensor_type=tensor_type)
            
        v = nppl.inputs.variational.IndNormal(
            torch.rand(timesamples, 1)*2*np.pi, torch.ones((timesamples, 1))*0.1, 'torus', d_z, tensor_type=tensor_type)
        
        latents = [nppl.inference.prior_variational_pair(_z, p, v)]
        
    elif z_mode[:2] == 'GP':
        d_z = int(z_mode[2:])
        latents = [temporal_GP(tensor_type=tensor_type)]
        
    elif z_mode == '':
        d_z = 0
        latents = []
        
    else:
        raise ValueError
        
    return latents, d_z



### statistics
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
        f_p = lambda c, avg, shape, t: utils.stats.poiss_count_prob(c, avg, shape, t)
    elif ll_mode[:2] == 'NB':
        if glm.dispersion_model is None:
            shape_model = glm.likelihood.r_inv.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)
    elif ll_mode[:3] == 'CMP':
        if glm.dispersion_model is None:
            shape_model = glm.likelihood.nu.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)
    elif ll_mode[:3] == 'ZIP':
        if glm.dispersion_model is None:
            shape_model = glm.likelihood.alpha.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
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
                    
        q_.append(utils.stats.count_KS_method(f_p, m_f, tbin, spktr, rm, shape=sh))
        
    if mode == 'single':
        cnt_tuples = [(q, *utils.stats.KS_statistics(q)) for q in q_]
        I = (rate_model*np.log(rate_model/rate_model.mean(-1, keepdims=True))).mean(-1)
    elif mode == 'population':
        q_tot = np.concatenate(q_)
        cnt_tuples = (q_tot, *utils.stats.KS_statistics(q_tot))
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
        t_spike = utils.neural.BinToTrain(spktrain[ne])
        q = utils.stats.ISI_KS_method(dist_isi, tbin, t_spike, rate_model)
        isi_tuples.append((q, *utils.stats.KS_statistics(q)))

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
        t_spike = utils.neural.BinToTrain(spktrain[ne])
        q = utils.stats.invISI_KS_method(dist_invisi, tbin, t_spike, rate_model)
        invisi_tuples.append((q, *utils.stats.KS_statistics(q)))

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
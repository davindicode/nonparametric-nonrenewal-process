import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
sys.path.append("../../neuroppl/")

import neuroppl as nppl
from neuroppl import base
from neuroppl import utils






class monotonic_GP_flows(base._input_mapping):
    """
    monotonic GP flows
    """
    def __init__(self, out_dims, num_induc, tx, ty, dt=1e-3, t_steps=100, interpolation='cubic', tensor_type=torch.float):
        """
        """
        Xu = torch.tensor(
            [np.linspace(tx.min().item(), tx.max().item(), num_induc)]
        ).T[None, ...].repeat(out_dims, 1, 1)
        in_dims = Xu.shape[-1]
        super().__init__(in_dims, out_dims, tensor_type, MC_only=True)
        
        ### interpolator ###
        self.register_buffer('tx', tx.type(tensor_type))
        self.register_buffer('ty', ty.type(tensor_type))
        
        if interpolation == 'cubic':
            self.interpolation_f = lambda X: utils.signal.cubic_interpolation(self.tx, self.ty, X, integrate=False)
        elif interpolation == 'linear':
            self.interpolation_f = lambda X: utils.signal.cubic_interpolation(self.tx, self.ty, X, integrate=False)
        else:
            raise ValueError('Interpolation mode not supported')

        ### GP ###
        v = 1.0*torch.ones(out_dims)
        l = (tx.max()-tx.min())/10.*torch.ones(1, out_dims)
        krn1 = nppl.kernels.kernel.Constant(variance=v, tensor_type=tensor_type)
        krn2 = nppl.kernels.kernel.RBF(
            input_dims=len(l), lengthscale=l, \
            track_dims=[0], topology='euclid', f='exp', \
            tensor_type=tensor_type
        )
        kernelobj = nppl.kernels.kernel.Product(krn1, krn2)
        inducing_points = nppl.kernels.kernel.inducing_points(out_dims, Xu, constraints=[])
        
        self.GP = nppl.mappings.SVGP(
            in_dims, out_dims, kernelobj, inducing_points=inducing_points, 
            whiten=True, jitter=1e-5, mean=0., learn_mean=False
        )
        
        ### integrator ###
        self.dt = dt
        self.t_steps = t_steps
        
    def KL_prior(self):
        return self.GP.KL_prior()
    
    def constrain(self):
        self.GP.constrain()
    
    def sample_F(self, XZ, eps=None):
        """
        Samples from the variational posterior.
        If time is outside self.tx range, we just get constant y output equivalent to a hard time boundary.
        """
        # input
        Ts = XZ.shape[-2]
        mc = XZ.shape[-4]
        
        if eps is None:
            eps = torch.randn(XZ.shape[:-1], dtype=self.tensor_type, device=XZ.device)
        print(XZ.min())
        print(XZ.max())
        # sample functions
        X = XZ # initial
        for t in range(self.t_steps):
            X = X + self.dt*self.GP.sample_F(X, eps)[..., None]
        
        return self.interpolation_f(X.flatten()).view(*X.shape[:-1])
    
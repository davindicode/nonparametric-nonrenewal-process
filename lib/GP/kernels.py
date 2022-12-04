from functools import partial


import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, tree_map, lax

from jax.scipy.linalg import expm, block_diag

from ..utils.jax import softplus, softplus_inv, softplus_inv_list, softplus_list
from ..utils.linalg import rotation_matrix, solve, solve_continuous_lyapunov

import scipy

import math
_sqrt_twopi = math.sqrt(2 * math.pi)




### functions ###
def broadcasting_elementwise(op, a, b):
    """
    Adapted from GPflow: https://github.com/GPflow/GPflow
    http://www.apache.org/licenses/LICENSE-2.0
    Apply binary operation `op` to every pair in tensors `a` and `b`.
    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(jnp.reshape(a, [-1, 1]), jnp.reshape(b, [1, -1]))
    return flatres.reshape(a.shape[0], b.shape[0])


### functions ###
def _safe_sqrt(x):
    # Clipping around the (single) float precision which is ~1e-45.
    return jnp.sqrt(jnp.maximum(x, 1e-36))


def _scaled_dist_squared_Rn(X, X_, lengthscale):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|^2`
    
    :param jnp.array X: input of shape (out_dims, num_points, in_dims)
    :return: distance of shape (out_dims, num_points, num_points_)
    """
    scaled_X = X / lengthscale[:, None, :]
    scaled_X_ = X_ / lengthscale[:, None, :]
    X2 = (scaled_X ** 2).sum(-1, keepdims=True)
    X2_ = (scaled_X_ ** 2).sum(-1, keepdims=True)
    XX_ = scaled_X @ scaled_X_.transpose(0, 2, 1)
    r2 = X2 - 2 * XX_ + X2_.transpose(0, 2, 1)
    return jnp.maximum(r2, 0.)


def _scaled_dist_Rn(X, Z, lengthscale):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|`
    """
    return jnp.sqrt(_scaled_dist_squared_Rn(X, Z, lengthscale))





### classes ###
class StationaryKernel(object):
    """
    The GP Kernel / prior class.
    Implements methods for converting GP priors,
        f(t) ~ GP(0,k(t,t'))
    into state space models.
    Constructs a linear time-invariant (LTI) stochastic differential
    equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(tₙ)=H x(tₙ))
    where w(t) is a white noise process and where the state x(t) is
    Gaussian distributed with initial state distribution x(t)~𝓝(0,Pinf).
    
    F      - Feedback matrix
    L      - Noise effect matrix
    Qc     - Spectral density of white noise process w(t)
    H      - Observation model matrix
    Pinf   - Covariance of the stationary process
    """
    def __init__(self, in_dims, out_dims, state_dims, hyp):
        self.hyp = hyp
        self.state_dims = state_dims  # state dynamics dimensions
        self.out_dims = out_dims  # dimensions after readout with H
        self.in_dims = in_dims  # input dimensions (1 for temporal)
        
    ### kernel ###
    def K(self, X, X_, hyp):
        """
        """
        raise NotImplementedError('Kernel function not implemented for this prior')
        
        
    ### state space ###
    def state_dynamics(self, hyp=None):
        """
        Returns F, L, Qc
        """
        raise NotImplementedError('Kernel to state space mapping not implemented for this prior')
        
    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt).
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters [array]
        :return: state transition matrix A [sd, sd]
        """
        F, _, _, _, _ = self.kernel_to_state_space(hyp)
        A = expm(F * dt)
        return A
    
    def state_output(self, hyp=None):
        """
        Returns H, minf, Pinf
        """
        raise NotImplementedError('Kernel to state space mapping not implemented for this prior')

    def sample_spectrum(self, hyp, prng_state, num_samps):
        """
        """
        raise NotImplementedError('Spectrum density is not implemented')
        
    @partial(jit, static_argnums=(0,))
    def spectral_representation(self, omega, hyp=None):
        """
        Calculation of the spectral representation of the kernel, from its state space parameters. 
        We return HS(\omega)H^T which is a scalar (directly in observation space.
        Note that in the LEG case this expression simplifies because of the structue of the process 
        (it assumes F = NN^T + R - R^T where N is the noise matrix)
        """
        F, L, Qc, H, _ = self.kernel_to_state_space(hyp)
        n = jnp.shape(F)[0]
        tmp = F+1j*jnp.eye(n)*omega
        tmp_inv = jnp.linalg.inv(tmp)
        conj_tmp_inv = jnp.linalg.inv(F-1j*jnp.eye(n)*omega)
        return H @ tmp_inv @ L @ Qc @ L.T @ conj_tmp_inv.T @ H.T

    @partial(jit, static_argnums=(0,))
    def temporal_representation(self, tau, hyp=None):
        """
        Calculation of the temporal representation of the kernel, from its state space parameters. 
        We return H P_inf*expm(F\tau)^TH^T
        """
        F, _, _, H, P_inf = self.kernel_to_state_space(hyp)
        return H @ P_inf @ expm(F.T*tau) @ H.T

    
    
    
class IID(StationaryKernel):
    """
    GPLVM latent space, i.e. no dynamics
    """
    def __init__(self, Qc):
        out_dims = Qc.shape[0]
        super().__init__(in_dims=1, out_dims=out_dims, state_dims=out_dims, hyp={'Qc': Qc})
        
    @partial(jit, static_argnums=(0,))
    def state_dynamics(self, hyp):
        hyp = self.hyp if hyp is None else hyp
        Qc = hyp['Qc']
        F = []
        L = []
        Pinf = Qc
        return F, L, Qc

    @partial(jit, static_argnums=(0,))
    def state_output(self, hyp):
        hyp = self.hyp if hyp is None else hyp
        H = jnp.ones(self.out_dims)
        minf = jnp.zeros((1,))  # stationary state mean
        Pinf = hyp['Qc']
        return H, minf, Pinf

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Cosine prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :param hyperparams: hyperparameters of the prior: frequency [1, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        return jnp.zeros((self.state_dims, self.state_dims))
        
    
    
class Cosine(StationaryKernel):
    """
    Cosine kernel in SDE form.
    Hyperparameters:
        radial frequency, ω
    The associated continuous-time state space model matrices are:
    F      = ( 0   -ω
               ω    0 )
    L      = N/A
    Qc     = N/A
    H      = ( 1  0 )
    Pinf   = ( 1  0
               0  1 )
    and the discrete-time transition matrix is (for step size Δt),
    A      = ( cos(ωΔt)   -sin(ωΔt)
               sin(ωΔt)    cos(ωΔt) )
    """
    def __init__(self, frequency):
        super().__init__(in_dims=1, out_dims=1, state_dims=2, hyp={'frequency': frequency})

    @property
    def frequency(self):
        return softplus(self.hyp['frequency'])
    
    @partial(jit, static_argnums=(0,))
    def state_dynamics(self, hyp):
        hyperparams = softplus(self.hyp if hyp is None else hyp)
        omega = hyperparams[0]
        F = jnp.array([[0.0,   -omega],
                      [omega, 0.0]])
        L = []
        Qc = []
        return F, L, Qc

    @partial(jit, static_argnums=(0,))
    def state_output(self, hyp):
        H = jnp.array([1.0, 0.0])
        minf = jnp.zeros((1,))  # stationary state mean
        Pinf = jnp.eye(H.shape[1])
        return H, minf, Pinf

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Cosine prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :param hyperparams: hyperparameters of the prior: frequency [1, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        hyperparams = softplus(self.hyp if hyp is None else hyp)
        omega = hyperparams[0]
        return rotation_matrix(dt, omega)  # [2, 2]
    
    
    
    


class LEG(StationaryKernel):
    """
    State-space formulation of the LEG model. The SDE takes the form 
    dz = -G/2zdt + Ndw where G = NN^T + R - R^T
    """
    def __init__(self, N, R, B, Lam):
        out_dims = B.shape[-1]
        state_dims = N.shape[0]
        in_dims = 1
        super().__init__(in_dims, out_dims, state_dims, 
                         hyp={'N': N, 'R': R, 'B': B, 'Lam': Lam})
        self.minf = jnp.zeros((state_dims,))
        
    @partial(jit, static_argnums=(0,))
    def parameterize(self, N, R):
        # symmetric part
        Q = jnp.diag(softplus(N) + 1e-5)
        #Q = N@N.T
        # antisymmetric part
        S = R - R.T 
        G = Q + S
        return G, Q 

    @partial(jit, static_argnums=(0,))
    def state_dynamics(self, hyperparams):
        N, R = hyperparams[0], hyperparams[1]
        G, Q = self.parameterize(N, R)
        F = -G/2.
        return F, Q

    @partial(jit, static_argnums=(0,))
    def state_output(self, hyperparams):
        F, Q = self.state_dynamics(hyperparams)
        H = hyperparams[2].T
        #Pinf is the solution of the Lyapunov equation F P + P F^T + L Qc L^T = 0
        # Pinf = solve_continuous_lyapunov(F, Q)
        # In this parameterization Pinf is just the identity
        Pinf = jnp.eye(H.shape[1])
        return H, self.minf, Pinf

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-7/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [4, 4]
        """
        # hyperparams = self.hyp if hyperparams is None else hyperparams
        hyperparams = self.hyp if hyperparams is None else hyperparams
        N, R =  hyperparams[0], hyperparams[1]
        G, _ = self.parameterize(N, R)
        return expm(-dt*G/2)
    
    @staticmethod
    def initialize_hyperparams(key, state_dims, out_dims):
        keys = jr.split(key,4)
        N = jnp.ones(state_dims) 
        R = jnp.eye(state_dims)
        B = jr.normal(keys[2], shape=(state_dims, out_dims))  / jnp.sqrt(state_dims)
        Lam = jr.normal(keys[3], shape=(state_dims,state_dims)) / jnp.sqrt(state_dims)
        return N, R, B, Lam
    
    def temporal_representation(self, tau, hyperparams=None):
        """
        Calculation of the temporal representation of the kernel, from its state space parameters. We return H P_inf*expm(F\tau)^TH^T
        """
        hyperparams = self.hyp if hyperparams is None else hyperparams
        F, Q = self.state_dynamics(hyperparams)
        H, _, P_inf = self.state_output(hyperparams)
        cond = jnp.sum(tau) >= 0. 
        def pos_tau():
            return H @ P_inf @ expm(tau*F.T)@ H.T
        def neg_tau():
            return H @ P_inf @ expm(jnp.abs(tau)*F.T)@ H.T
        return lax.cond(cond, pos_tau, neg_tau)
    #return H @ P_inf @ expm(tau*F) @ H.T
    
    
    
### lengthscale ###
class Lengthscale(StationaryKernel):
    """
    Kernels based on lengthscales
    """
    def __init__(self, out_dims, state_dims, variance, lengthscale, distance_metric):
        hyp = {'lengthscale': softplus_inv(lengthscale), 
               'variance': softplus_inv(variance)}
        in_dims = lengthscale.shape[-1]
        super().__init__(in_dims, out_dims, state_dims, hyp=hyp)
        self.distance_metric = distance_metric
    
    @property
    def variance(self):
        return softplus(self.hyp['variance'])

    @property
    def lengthscale(self):
        return softplus(self.hyp['lengthscale'])
    
    # kernel
    @partial(jit, static_argnums=(0,))
    def K(self, X, X_, hyp=None):
        """
        X and X_ have shapes (out_dims, num_points, in_dims)
        """
        hyp = self.hyp if hyp is None else hyp
        
        l = softplus(hyp['lengthscale'])
        r_in = self.distance_metric(X, X_, l)
        
        #hyp_no_l = {k: hyp[k] for k in list(hyp)[1:]}
        return self.K_r(r_in, hyp)
    
    @staticmethod
    def K_r(r, hyp):
        raise NotImplementedError('kernel not implemented')
    
    @staticmethod
    def K_omega(r, hyp):
        raise NotImplementedError('kernel not implemented')

    # state space
    @partial(jit, static_argnums=(0,))
    def state_dynamics(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        in_shape = tree_map(lambda x: 0, hyp)
        
        # vmap over output dimensions
        out = vmap(
            self.state_dynamics_, 
            in_axes=in_shape, 
            out_axes=(0,)*3
        )(hyp)
        return block_diag(*out)
    
    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        in_shape = tree_map(lambda x: 0, hyp)
        
        # vmap over output dimensions
        out = vmap(
            self.state_transition_, 
            in_axes=(None, in_shape), 
            out_axes=0
        )(dt, hyp)
        
        return block_diag(*out)
        
    @partial(jit, static_argnums=(0,))
    def state_output(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        in_shape = tree_map(lambda x: 0, hyp)
        # vmap over output dimensions
        H, minf, Pinf = vmap(
            self.state_output_, 
            in_axes=(in_shape,), 
            out_axes=(0,)*3         
        )(hyp)
        return block_diag(*H), minf.reshape(-1), block_diag(*Pinf)
    
    
    
    
class Matern12(Lengthscale):
    """
    Exponential, i.e. Matern-1/2 kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    F      = -1/l
    L      = 1
    Qc     = 2σ²/l
    H      = 1
    Pinf   = σ²
    """
    def __init__(self, out_dims, variance, lengthscale):
        state_dims = out_dims
        super().__init__(out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn)

    @staticmethod
    def K_r(r, hyp):
        """
        The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
        differentiable anywhere. The kernel equation is
        k(r) = σ² exp{-r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
        σ² is the variance parameter
        """
        variance = softplus(hyp['variance'])[:, None, None]
        return variance * jnp.exp(-r)
    
        
    @partial(jit, static_argnums=(0,))
    def state_dynamics_(self, hyp):
        """
        Uses variance and lengthscale hyperparameters to construct the state space model
        """
        hyp = self.hyp if hyp is None else hyp
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        F = -1.0 / ell[None, None]
        L = jnp.ones((1, 1))
        Qc = 2.0 * (var / ell)[None, None]
        return F, L, Qc

    @partial(jit, static_argnums=(0,))
    def state_transition_(self, dt, hyp):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [1, 1]
        """
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        A = jnp.exp(-dt / ell)[None, None]
        return A

    @partial(jit, static_argnums=(0,))
    def state_output_(self, hyp):
        var = softplus(hyp['variance'])
        #F, L, Qc = self.state_dynamics(hyp)
        H = jnp.ones((1, 1)) # observation projection
        minf = jnp.zeros((1,))  # stationary state mean
        Pinf = var[None, None]
        return H, minf, Pinf
    
    

class Matern32(Lengthscale):
    """
    Matern-3/2 kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    letting λ = √3/l
    F      = ( 0   1
              -λ² -2λ)
    L      = (0
              1)
    Qc     = 4λ³σ²
    H      = (1  0)
    Pinf   = (σ²  0
              0   λ²σ²)
    """
    def __init__(self, out_dims, variance, lengthscale):
        state_dims = 2*out_dims
        super().__init__(out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn)

    
    @staticmethod
    def K_r(r, hyp):
        """
        The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
        differentiable. The kernel equation is
        k(r) = σ² (1 + √3r) exp{-√3 r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
        σ² is the variance parameter.
        """
        variance = softplus(hyp['variance'])[:, None, None]
        sqrt3 = jnp.sqrt(3.0)
        return variance * (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r)
        
    @partial(jit, static_argnums=(0,))
    def state_dynamics_(self, hyp):
        hyp = self.hyp if hyp is None else hyp
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        lam = 3.0 ** 0.5 / ell
        F = jnp.array([[0.0,       1.0     ],
                      [-lam ** 2, -2 * lam]])
        L = jnp.array([[0],
                      [1]])
        Qc = jnp.array([[12.0 * 3.0 ** 0.5 / ell ** 3.0 * var]])
        return F, L, Qc

    @partial(jit, static_argnums=(0,))
    def state_transition_(self, dt, hyp):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [2, 2]
        """
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        lam = jnp.sqrt(3.0) / ell
        A = jnp.exp(-dt * lam) * (dt * jnp.array([[lam, 1.0], [-lam**2.0, -lam]]) + jnp.eye(2))
        return A

    @partial(jit, static_argnums=(0,))
    def state_output_(self, hyp):
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        #print(ell.shape, var.shape, "tree")
        #F, L, Qc = self.state_dynamics(hyp)
        H = jnp.array([[1.0, 0.0]]) # observation projection
        minf = jnp.zeros((2,))  # stationary state mean
        Pinf = jnp.array([[var, 0.0],
                         [0.0, 3.0 * var / ell ** 2.0]])
        return H, minf, Pinf

    
class Matern52(Lengthscale):
    """
    Matern-5/2 kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    letting λ = √5/l
    F      = ( 0    1    0
               0    0    1
              -λ³ -3λ² -3λ)
    L      = (0
              0
              1)
    Qc     = 16λ⁵σ²/3
    H      = (1  0  0)
    letting κ = λ²σ²/3,
    Pinf   = ( σ²  0  -κ
               0   κ   0
              -κ   0   λ⁴σ²)
    """
    def __init__(self, out_dims, variance, lengthscale):
        state_dims = 3*out_dims
        super().__init__(out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn)
        
    @staticmethod
    def K_r(r, hyp):
        """
        The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
        differentiable. The kernel equation is
        k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
        σ² is the variance parameter.
        """
        var = softplus(hyp['variance'])[:, None, None]
        sqrt5 = jnp.sqrt(5.0)
        return var * (1.0 + sqrt5 * r + 5.0 / 3.0 * jnp.square(r)) * jnp.exp(-sqrt5 * r)

    @partial(jit, static_argnums=(0,))
    def state_dynamics_(self, hyp):
        hyp = self.hyp if hyp is None else hyp
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        lam = 5.0**0.5 / ell
        F = jnp.array([[0.0,       1.0,           0.0     ],
                      [0.0,       0.0,           1.0     ],
                      [-lam**3.0, -3.0*lam**2.0, -3.0*lam]])
        L = jnp.array([[0.0],
                      [0.0],
                      [1.0]])
        Qc = jnp.array([[var * 400.0 * 5.0 ** 0.5 / 3.0 / ell ** 5.0]])
        return F, L, Qc

    @partial(jit, static_argnums=(0,))
    def state_transition_(self, dt, hyp):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [3, 3]
        """
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        lam = jnp.sqrt(5.0) / ell
        dtlam = dt * lam
        A = jnp.exp(-dtlam) \
            * (dt * jnp.array([[lam * (0.5 * dtlam + 1.0),      dtlam + 1.0,            0.5 * dt                 ],
                              [-0.5 * dtlam * lam ** 2,        lam * (1.0 - dtlam),    1.0 - 0.5 * dtlam        ],
                              [lam ** 3 * (0.5 * dtlam - 1.0), lam ** 2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)]])
               + jnp.eye(3))
        return A
    
    @partial(jit, static_argnums=(0,))
    def state_output_(self, hyp):
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        #F, L, Qc = self.state_dynamics(hyp)
        H = jnp.array([[1.0, 0.0, 0.0]]) # observation projection
        minf = jnp.zeros((3,))  # stationary state mean
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf = jnp.array([[var,    0.0,   -kappa],
                         [0.0,    kappa, 0.0],
                         [-kappa, 0.0,   25.0*var / ell**4.0]])
        return H, minf, Pinf


    
class Matern72(Lengthscale):
    """
    Matern-7/2 kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    letting λ = √7/l
    F      = ( 0    1    0    0
               0    0    1    0
               0    0    0    1
              -λ⁴ -4λ³ -6λ²  -4λ)
    L      = (0
              0
              0
              1)
    Qc     = 10976σ²√7/(5l⁷)
    H      = (1  0  0  0)
    letting κ = λ²σ²/5,
    and    κ₂ = 72σ²/l⁴
    Pinf   = ( σ²  0  -κ   0
               0   κ   0  -κ₂
               0  -κ₂  0   343σ²/l⁶)
    """
    def __init__(self, out_dims, variance, lengthscale):
        state_dims = 4*out_dims
        super().__init__(out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn)

    # kernel
    @staticmethod
    def K_r(r, hyp):
        var = softplus(hyp['variance'])[:, None, None]
        sqrt7 = jnp.sqrt(7.0)
        return variance * (
            1. + sqrt7 * r + 14. / 5. * jnp.square(r) + 7. * sqrt7 / 15. * r**3
        ) * jnp.exp(-sqrt7 * r)
        
    # state space
    @partial(jit, static_argnums=(0,))
    def state_dynamics_(self, hyp):     
        hyp = self.hyp if hyp is None else hyp
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        lam = 7.0**0.5 / ell
        F = jnp.array([[0.0,       1.0,           0.0,           0.0     ],
                      [0.0,       0.0,           1.0,           0.0     ],
                      [0.0,       0.0,           0.0,           1.0     ],
                      [-lam**4.0, -4.0*lam**3.0, -6.0*lam**2.0, -4.0*lam]])
        L = jnp.array([[0.0],
                      [0.0],
                      [0.0],
                      [1.0]])
        Qc = jnp.array([[var * 10976.0 * 7.0 ** 0.5 / 5.0 / ell ** 7.0]])
        
        return F, L, Qc
        
    @partial(jit, static_argnums=(0,))
    def state_transition_(self, dt, hyp=None):
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        lam = jnp.sqrt(7.0) / ell
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam ** 2
        A = jnp.exp(-dtlam) \
            * (dt * jnp.array([[lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),      1.0 + dtlam + 0.5 * dtlam2,
                              0.5 * dt * (1.0 + dtlam),                       dt ** 2 / 6],
                              [-dtlam2 * lam ** 2.0 / 6.0,                    lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                              1.0 + dtlam - 0.5 * dtlam2,                     dt * (0.5 - dtlam / 6.0)],
                              [lam3 * dtlam * (dtlam / 6.0 - 0.5),            dtlam * lam2 * (0.5 * dtlam - 2.0),
                              lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),       1.0 - dtlam + dtlam2 / 6.0],
                              [lam2 ** 2 * (dtlam - 1.0 - dtlam2 / 6.0),      lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                              lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),      lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0)]])
               + jnp.eye(4))
        return A
        

    @partial(jit, static_argnums=(0,))
    def state_output_(self, hyp):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-7/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [4, 4]
        """
        var = softplus(hyp['variance'])
        ell = softplus(hyp['lengthscale'][0]) # use first input dimension as time dimension
        
        #F, L, Qc = self.state_dynamics(hyp)
        H = jnp.array([[1.0, 0.0, 0.0, 0.0]]) # observation projection
        minf = jnp.zeros((4,))  # stationary state mean
        kappa = 7.0 / 5.0 * var / ell**2.0
        kappa2 = 9.8 * var / ell**4.0
        
        Pinf = jnp.array([[var,    0.0,     -kappa, 0.0                 ],
                         [0.0,    kappa,   0.0,    -kappa2             ],
                         [-kappa, 0.0,     kappa2, 0.0                 ],
                         [0.0,    -kappa2, 0.0,    343.0*var / ell**6.0]])
        return H, minf, Pinf
    


    
    
    
class SquaredExponential(Lengthscale):
    r"""
    Implementation of Squared Exponential kernel:

        :math:`k(x,z) = \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    The classic square exponential kernel. Functions drawn from a GP with this kernel are infinitely
    differentiable. The kernel equation is
    k(r) = σ² exp{-r^2}
    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ² is the variance parameter
    """
    def __init__(self, out_dims, variance, lengthscale):
        super().__init__(out_dims, None, variance, lengthscale, _scaled_dist_squared_Rn)
        
    @staticmethod
    def K_r(r2, hyp):
        variance = softplus(hyp['variance'])[:, None, None]
        return variance * jnp.exp(-0.5 * r2)

    @staticmethod
    def K_omega(omega, hyp):
        lengthscale, variance = hyp
        return variance * _sqrt_twopi * lengthscale * jnp.exp(-2 * omega**2 * lengthscale**2)
    
    def sample_spectrum(self, hyp, prng_state, num_samps, RFF_num_feats):
        lengthscale = softplus(hyp['lengthscale'])  # (out, d_x)
        k_std = 1. / lengthscale
        ks = k_std[None, :, None, :] * jr.normal(
            prng_state, shape=(num_samps, self.out_dims, RFF_num_feats, self.in_dims))
        return ks

    @partial(jit, static_argnums=0)
    def state_transition_(self, dt, hyp=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [1, 1]
        """
        hyperparams = softplus(self.hyp if hyp is None else hyp)
        ell = hyperparams[1]
        
        A = jnp.exp(-dt / ell)[None, None]
        return A



class RationalQuadratic(Lengthscale):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param jnp.array scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """
    def __init__(self, out_dims, variance, lengthscale, scale_mixture):
        super().__init__(out_dims, None, variance, lengthscale, _scaled_dist_squared_Rn)
        self.hyp += [softplus_inv(scale_mixture)]
        
    @property
    def scale_mixture(self):
        return self.lf(self._scale_mixture)[None, :, None] # K, N, T
    
    @scale_mixture.setter
    def scale_mixture(self):
        self._scale_mixture.data = self.lf_inv(scale_mixture)

    
    @staticmethod
    def K_r(r2, hyp):
        variance = softplus(hyp['variance'])[:, None, None]
        scale_mixture = softplus(hyp['scale_mixture'])[:, None, None]
        return variance * (1 + (0.5 / scale_mixture) * r2).pow(-scale_mixture)


    
    

    
### sinusoidal ###
class HidaMaternKernel12(StationaryKernel):
    """
    The Hida Matern kernel (Dowling, Sokol, Park 2021; https://arxiv.org/pdf/2107.07098.pdf) 
    For this kernel, we need to specify the order of the Matern kernel it's constructed from (here, 1/2).
    Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is
    k(r) = σ² cos(r*b) exp{-r} where we reused StationaryKernel class which was defined in terms of spatial components, such that r here has the interpreration tau/l.
    """
    @staticmethod
    def K_r(r, lengthscale = 1., b=1., variance = 1.):
        return variance * jnp.cos(b*r) * jnp.exp(-r/lengthscale)

class HidaMaternKernel32(StationaryKernel):
    @staticmethod
    def K_r(r, lengthscale = 1., b=1., variance = 1.):
        sqrt3 = jnp.sqrt(3.0)
        return variance * jnp.cos(b*r) * (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r/lengthscale)


class HidaMaternKernel52(StationaryKernel):
    @staticmethod
    def K_r(r, lengthscale = 1., b=1., variance = 1.):
        sqrt5 = jnp.sqrt(5.0)
        return variance * jnp.cos(b*r)* (1.0 + sqrt5 * r + 5.0 / 3.0 * jnp.square(r)) * jnp.exp(-sqrt5 * r)


    
    
    
### subband ###
class SubbandLengthscale(StationaryKernel):
    """
    State space equivalent of matern cosine product
    """
    def __init__(self, out_dims, state_dims, variance, lengthscale, radial_frequency, in_dims=1):
        hyp = {'lengthscale': lengthscale, 'variance': variance, 'radial_frequency': radial_frequency}
        super().__init__(in_dims, out_dims, state_dims, hyp=hyp)
        
    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])
        
    @property
    def radial_frequency(self):
        return softplus(self.hyp[2])

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        in_shape = tree_map(lambda x: 0, hyp)
        
        # vmap over output dimensions
        out = vmap(
            self.state_transition_, 
            in_axes=(None, in_shape), 
            out_axes=0
        )(dt, hyp)
        A = block_diag(*out)
        return A
        
    @partial(jit, static_argnums=(0,))
    def state_output(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        in_shape = tree_map(lambda x: 0, hyp)
        # vmap over output dimensions
        H, minf, Pinf = vmap(
            self.state_output_, 
            in_axes=(in_shape,), 
            out_axes=(0,)*3         
        )(hyp)
        return block_diag(*H), minf.reshape(-1), block_diag(*Pinf)
    
    
    

class SubbandMatern12(SubbandLengthscale):
    """
    Subband Matern-1/2 (i.e. Exponential) kernel in SDE form (product of Cosine and Matern-1/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the exponential and cosine components:
    F      = F_exp ⊕ F_cos  =  ( -1/l  -ω
                                 ω     -1/l )
    L      = L_exp ⊗ I      =  ( 1      0
                                 0      1 )
    Qc     = I ⊗ Qc_exp     =  ( 2σ²/l  0
                                 0      2σ²/l )
    H      = H_exp ⊗ H_cos  =  ( 1      0 )
    Pinf   = Pinf_exp ⊗ I   =  ( σ²     0
                                 0      σ² )
    and the discrete-time transition matrix is (for step size Δt),
    A      = exp(-Δt/l) ( cos(ωΔt)   -sin(ωΔt)
                          sin(ωΔt)    cos(ωΔt) )
    """
    def __init__(self, variance, lengthscale, radial_frequency, out_dims, in_dims=1):
        state_dims=2*out_dims
        super().__init__(out_dims, state_dims, variance, lengthscale, radial_frequency, in_dims)
        H_mat = jnp.array([[1.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        self.H = jnp.kron(H_mat, H_cos)


    @partial(jit, static_argnums=(0,))
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, omega = hyperparams
        F_mat = jnp.array([[-1.0 / ell]])
        L_mat = jnp.array([[1.0]])
        Qc_mat = jnp.array([[2.0 * var / ell]])
        Pinf_mat = jnp.array([[var]])
        F_cos = jnp.array([[0.0, -omega],
                          [omega, 0.0]])
        # F = (-1/l -ω
        #      ω    -1/l)
        F = jnp.kron(F_mat, jnp.eye(2)) + F_cos
        L = jnp.kron(L_mat, jnp.eye(2))
        Qc = jnp.kron(jnp.eye(2), Qc_mat)
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return F, L, Qc, Pinf

    @partial(jit, static_argnums=(0,))
    def state_transition_(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-1/2 prior:
        A = exp(-Δt/l) ( cos(ωΔt)   -sin(ωΔt)
                         sin(ωΔt)    cos(ωΔt) )
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, frequency [3, 1]
        :return: state transition matrix A [2, 2]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, omega = softplus(hyperparams['variance']),  softplus(hyperparams['lengthscale'][0]),  softplus(hyperparams['radial_frequency'][0])
        R = rotation_matrix(dt, omega)
        A = jnp.exp(-dt / ell) * R  # [2, 2]
        return A

    @partial(jit, static_argnums=(0,))
    def state_output_(self, hyp):
        var, ell, omega = softplus(hyp['variance']),  softplus(hyp['lengthscale'][0]),  softplus(hyp['radial_frequency'][0])
        #F, L, Qc = self.state_dynamics(hyp)
        H_mat = jnp.array([[1.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        H = jnp.kron(H_mat, H_cos)
        minf = jnp.zeros((2,))  # stationary state mean
        Pinf_mat = jnp.array([[var]])
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return H, minf, Pinf
    




class SubbandMatern32(StationaryKernel):
    """
    Subband Matern-3/2 kernel in SDE form (product of Cosine and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the Matern3/2 and cosine components:
    letting λ = √3 / l
    F      = F_mat3/2 ⊕ F_cos  =  ( 0     -ω     1     0
                                    ω      0     0     1
                                   -λ²     0    -2λ   -ω
                                    0     -λ²    ω    -2λ )
    L      = L_mat3/2 ⊗ I      =  ( 0      0
                                    0      0
                                    1      0
                                    0      1 )
    Qc     = I ⊗ Qc_mat3/2     =  ( 4λ³σ²  0
                                    0      4λ³σ² )
    H      = H_mat3/2 ⊗ H_cos  =  ( 1      0     0      0 )
    Pinf   = Pinf_mat3/2 ⊗ I   =  ( σ²     0     0      0
                                    0      σ²    0      0
                                    0      0     3σ²/l² 0
                                    0      0     0      3σ²/l²)
    and the discrete-time transition matrix is (for step size Δt),
    R = ( cos(ωΔt)   -sin(ωΔt)
          sin(ωΔt)    cos(ωΔt) )
    A = exp(-Δt/l) ( (1+Δtλ)R   ΔtR
                     -Δtλ²R    (1-Δtλ)R )
    """

    def __init__(self, variance=1.0, lengthscale=1.0, radial_frequency=1.0, out_dims=1):
        hyp = [variance, lengthscale, radial_frequency]
        super().__init__(hyp=hyp)
        H_mat = jnp.array([[1.0, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        self.H = jnp.kron(H_mat, H_cos)

    #@partial(jit, static_argnums=(0,))
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, omega = hyperparams
        lam = 3.0 ** 0.5 / ell
        F_mat = jnp.array([[0.0, 1.0],
                          [-lam ** 2, -2 * lam]])
        L_mat = jnp.array([[0],
                          [1]])
        Qc_mat = jnp.array([[12.0 * 3.0 ** 0.5 / ell ** 3.0 * var]])
        H_mat = jnp.array([[1.0, 0.0]])
        Pinf_mat = jnp.array([[var, 0.0],
                             [0.0, 3.0 * var / ell ** 2.0]])
        F_cos = jnp.array([[0.0, -omega],
                          [omega, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = jnp.kron(F_mat, jnp.eye(2)) + jnp.kron(jnp.eye(2), F_cos)
        L = jnp.kron(L_mat, jnp.eye(2))
        Qc = jnp.kron(jnp.eye(2), Qc_mat)
        H = jnp.kron(H_mat, H_cos)
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return F, L, Qc, Pinf

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, frequency [3, 1]
        :return: state transition matrix A [4, 4]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, omega = hyperparams[1], hyperparams[2]
        lam = jnp.sqrt(3.0) / ell
        R = rotation_matrix(dt, omega)
        A = jnp.exp(-dt * lam) * jnp.block([
            [(1. + dt * lam) * R, dt * R],
            [-dt * lam ** 2 * R, (1. - dt * lam) * R]
        ])
        return A



### combinations ###
class Independent(StationaryKernel):
    """
    A stack of independent GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models such that each component is fed to the likelihood.
    This class differs from Sum only in the measurement model.
    """
    def __init__(self, components):
        hyp = []
        out_dims = 0
        state_dims = 0
        in_dims = components[0].in_dims
        for c in components:
            hyp += [c.hyp]
            out_dims += c.out_dims
            state_dims += c.state_dims
            if in_dims != c.in_dims:
                raise ValueError('Input dimensions must be the same for all kernels')
            
        super().__init__(in_dims, out_dims, state_dims, hyp)
        self.components = components  
        
    # kernel
    @partial(jit, static_argnums=(0,))
    def K(self, X, X_, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        
        for i in range(1, len(self.components)):
            if i == 0: # use only variance of first kernel component
                variance = hyp[0][0]
            else:
                variance = 1.
            r_in = self.distance_metric(X, X_, hyp[i][1:])
            return self.K_r(r_in, variance)
    
        
    # state space
    @partial(jit, static_argnums=(0,))
    def state_dynamics(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        
        F, L, Qc = self.components[0].state_dynamics(hyp[0])
        for i in range(1, len(self.components)):
            F_, L_, Qc_ = self.components[i].state_dynamics(hyp[i])
            F = jnp.block([
                [F, jnp.zeros([F.shape[0], F_.shape[1]])],
                [jnp.zeros([F_.shape[0],   F.shape[1]]), F_]
            ])
            L = jnp.block([
                [L, jnp.zeros([L.shape[0], L_.shape[1]])],
                [jnp.zeros([L_.shape[0],   L.shape[1]]), L_]
            ])
            Qc = jnp.block([
                [Qc,                     jnp.zeros([Qc.shape[0], Qc_.shape[1]])],
                [jnp.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_]
            ])
        return F, L, Qc

    

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: [array]
        :return: state transition matrix A [D, D]
        """
        hyp = self.hyp if hyp is None else hyp
        A = self.components[0].state_transition(dt, hyp[0])
        for i in range(1, len(self.components)):
            A_ = self.components[i].state_transition(dt, hyp[i])
            A = jnp.block([
                [A, jnp.zeros([A.shape[0], A_.shape[0]])],
                [jnp.zeros([A_.shape[0], A.shape[0]]), A_]
            ])
        return A
    
    
    @partial(jit, static_argnums=(0,))
    def state_output(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        H, minf, Pinf = self.components[0].state_output(hyp[0])
        for i in range(1, len(self.components)):
            H_, minf_, Pinf_ = self.components[i].state_output(hyp[i])
            H = jnp.block([
                [H, jnp.zeros([H.shape[0], H_.shape[1]])],
                [jnp.zeros([H_.shape[0],   H.shape[1]]), H_]
            ])
            minf = jnp.concatenate([minf, minf_])
            Pinf = jnp.block([
                [Pinf, jnp.zeros([Pinf.shape[0],             Pinf_.shape[1]])],
                [jnp.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_]
            ])
            
        return H, minf, Pinf
    
    
    
class Independent(StationaryKernel):
    """
    A stack of independent GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models such that each component is fed to the likelihood.
    This class differs from Sum only in the measurement model.
    """
    def __init__(self, components):
        hyp = []
        out_dims = 0
        state_dims = 0
        in_dims = components[0].in_dims
        for c in components:
            hyp += [c.hyp]
            out_dims += c.out_dims
            state_dims += c.state_dims
            if in_dims != c.in_dims:
                raise ValueError('Input dimensions must be the same for all kernels')
            
        super().__init__(in_dims, out_dims, state_dims, hyp)
        self.components = components  
        
    # kernel
    @partial(jit, static_argnums=(0,))
    def K(self, X, X_, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        
        for i in range(1, len(self.components)):
            if i == 0: # use only variance of first kernel component
                variance = hyp[0][0]
            else:
                variance = 1.
            r_in = self.distance_metric(X, X_, hyp[i][1:])
            return self.K_r(r_in, variance)
    
        
    # state space
    @partial(jit, static_argnums=(0,))
    def state_dynamics(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        
        F, L, Qc = self.components[0].state_dynamics(hyp[0])
        for i in range(1, len(self.components)):
            F_, L_, Qc_ = self.components[i].state_dynamics(hyp[i])
            F = jnp.block([
                [F, jnp.zeros([F.shape[0], F_.shape[1]])],
                [jnp.zeros([F_.shape[0],   F.shape[1]]), F_]
            ])
            L = jnp.block([
                [L, jnp.zeros([L.shape[0], L_.shape[1]])],
                [jnp.zeros([L_.shape[0],   L.shape[1]]), L_]
            ])
            Qc = jnp.block([
                [Qc,                     jnp.zeros([Qc.shape[0], Qc_.shape[1]])],
                [jnp.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_]
            ])
        return F, L, Qc

    

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyp=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: [array]
        :return: state transition matrix A [D, D]
        """
        hyp = self.hyp if hyp is None else hyp
        A = self.components[0].state_transition(dt, hyp[0])
        for i in range(1, len(self.components)):
            A_ = self.components[i].state_transition(dt, hyp[i])
            A = jnp.block([
                [A, jnp.zeros([A.shape[0], A_.shape[0]])],
                [jnp.zeros([A_.shape[0], A.shape[0]]), A_]
            ])
        return A
    
    
    @partial(jit, static_argnums=(0,))
    def state_output(self, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        H, minf, Pinf = self.components[0].state_output(hyp[0])
        for i in range(1, len(self.components)):
            H_, minf_, Pinf_ = self.components[i].state_output(hyp[i])
            H = jnp.block([
                [H, jnp.zeros([H.shape[0], H_.shape[1]])],
                [jnp.zeros([H_.shape[0],   H.shape[1]]), H_]
            ])
            minf = jnp.concatenate([minf, minf_])
            Pinf = jnp.block([
                [Pinf, jnp.zeros([Pinf.shape[0],             Pinf_.shape[1]])],
                [jnp.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_]
            ])
            
        return H, minf, Pinf

    
    
class Product(object):
    """
    The product kernel for handling multi-dimensional input
    """
    def __init__(self, components):
        hyp = [components[0].hyp]
        for i in range(1, len(components)):
            hyp = hyp + [components[i].hyp]
        self.components = components
        self.hyp = hyp
    
    
    
class Sum(object):
    """
    A sum of GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models to produce their sum.
    """
    def __init__(self, components):
        hyp = [components[0].hyp]
        for i in range(1, len(components)):
            hyp = hyp + [components[i].hyp]
        self.components = components
        self.hyp = hyp

    @partial(jit, static_argnums=(0,))
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus_list(self.hyp) if hyperparams is None else hyperparams
        F, L, Qc, H, Pinf = self.components[0].kernel_to_state_space(hyperparams[0])
        for i in range(1, len(self.components)):
            F_, L_, Qc_, H_, Pinf_ = self.components[i].kernel_to_state_space(hyperparams[i])
            F = jnp.block([
                [F, jnp.zeros([F.shape[0], F_.shape[1]])],
                [jnp.zeros([F_.shape[0],   F.shape[1]]), F_]
            ])
            L = jnp.block([
                [L, jnp.zeros([L.shape[0], L_.shape[1]])],
                [jnp.zeros([L_.shape[0],   L.shape[1]]), L_]
            ])
            Qc = jnp.block([
                [Qc,                     jnp.zeros([Qc.shape[0], Qc_.shape[1]])],
                [jnp.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_]
            ])
            H = jnp.block([
                H, H_
            ])
            Pinf = jnp.block([
                [Pinf, jnp.zeros([Pinf.shape[0],             Pinf_.shape[1]])],
                [jnp.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_]
            ])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=(0,))
    def measurement_model(self, r=None, hyperparams=None):
        hyperparams = softplus_list(self.hyp) if hyperparams is None else hyperparams
        H = self.components[0].measurement_model(r, hyperparams[0])
        for i in range(1, len(self.components)):
            H_ = self.components[i].measurement_model(r, hyperparams[i])
            H = jnp.block([
                H, H_
            ])
        return H

    @partial(jit, static_argnums=(0,))
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: [array]
        :return: state transition matrix A [D, D]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        A = self.components[0].state_transition(dt, hyperparams[0])
        for i in range(1, len(self.components)):
            A_ = self.components[i].state_transition(dt, hyperparams[i])
            A = jnp.block([
                [A, jnp.zeros([A.shape[0], A_.shape[0]])],
                [jnp.zeros([A_.shape[0], A.shape[0]]), A_]
            ])
        return A
    
    def spectral_representation(self, omega, hyperparams=None):
        """
        Calculation of the spectral representation of the kernel, from its state space parameters. We return HS(\omega)H^T which is a scalar (directly in observation space.
        Note that in the LEG case this expression simplifies because of the structue of the process (it assumes F = NN^T + R - R^T where N is the noise matrix)
        """
        F, L, Qc, H, _ = self.kernel_to_state_space(hyperparams)
        n = jnp.shape(F)[0]
        tmp = F+1j*jnp.eye(n)*omega
        tmp_inv = jnp.linalg.inv(tmp)
        conj_tmp_inv = jnp.linalg.inv(F-1j*jnp.eye(n)*omega)
        return H@tmp_inv@L@Qc@L.T@conj_tmp_inv.T@H.T

    def temporal_representation(self, tau, hyperparams=None):
        """
        Calculation of the temporal representation of the kernel, from its state space parameters. We return H P_inf*expm(F\tau)^TH^T
        """
        F, _, _, H, P_inf = self.kernel_to_state_space(hyperparams)
        return H@P_inf@expm(F.T*tau)@H.T
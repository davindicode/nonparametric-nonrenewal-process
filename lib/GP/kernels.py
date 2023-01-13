import math
from typing import Callable, List

import equinox as eqx

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree_map, vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import expm, solve_triangular

from ..base import module

from ..utils.jax import safe_sqrt, softplus, softplus_inv
from ..utils.linalg import rotation_matrix, solve_continuous_lyapunov

from .linalg import bdiag, id_kronecker, LTI_process_noise

_sqrt_twopi = math.sqrt(2 * math.pi)



### functions ###
def _scaled_dist_squared_Rn(X, Y, lengthscale, diagonal):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|^2`

    :param jnp.array X: input of shape (out_dims, num_points, in_dims)
    :return: distance of shape (out_dims, num_points, num_points)
    """
    scaled_X = X / lengthscale[:, None, :]
    scaled_Y = Y / lengthscale[:, None, :]
    X2 = (scaled_X**2).sum(-1, keepdims=True)
    Y2 = (scaled_Y**2).sum(-1, keepdims=True)
    if diagonal:
        XY = (scaled_X * scaled_Y).sum(-1, keepdims=True)
        r2 = X2 - 2 * XY + Y2  # (out_dims, num_points, 1)
    else:
        XY = scaled_X @ scaled_Y.transpose(0, 2, 1)
        r2 = X2 - 2 * XY + Y2.transpose(0, 2, 1)
    return jnp.maximum(r2, 0.0)  # numerically threshold


def _scaled_dist_Rn(X, Y, lengthscale, diagonal):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|`
    """
    return safe_sqrt(_scaled_dist_squared_Rn(X, Y, lengthscale, diagonal))


### base classes ###
class Kernel(module):
    in_dims: int
    out_dims: int

    def __init__(self, in_dims, out_dims, array_type):
        super().__init__(array_type)
        self.in_dims = in_dims
        self.out_dims = out_dims

    ### kernel ###
    def K(self, X, Y, diagonal):
        """
        Kernel covariance function
        """
        raise NotImplementedError("Covariance function not implemented for this kernel")


class StationaryKernel(Kernel):
    """
    The stationary GP kernel class
        f(t) ~ GP(0,k(t,t')) = GP(0,k(t-t'))
    """

    ### Fourier domain ###
    def sample_spectrum(self, prng_state, num_samps):
        """ """
        raise NotImplementedError("Spectrum density is not implemented")


class MarkovianKernel(StationaryKernel):
    """
    The stationary GP kernel class
        f(t) ~ GP(0,k(t,t'))
    with a linear time-invariant (LTI) stochastic differential
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

    state_dims: int

    def __init__(self, in_dims, out_dims, state_dims, array_type):
        """
        dimensions after readout with H
        input dimensions (1 for temporal)
        """
        super().__init__(in_dims, out_dims, array_type)
        self.state_dims = state_dims  # state dynamics dimensions

    ### state space ###
    def _state_dynamics(self):
        """
        Returns F, L, Qc
        """
        raise NotImplementedError(
            "State space dynamics not implemented for this kernel"
        )

    def _state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt).

        :param dt: scalar step size, Δtₙ = tₙ - tₙ₋₁
        :return: state transition matrix A (state_dims, state_dims,)
        """
        raise NotImplementedError(
            "State space transition not implemented for this kernel"
        )

    def _state_output(self):
        """
        Returns H, Pinf
        """
        raise NotImplementedError(
            "State space matrices not implemented for this kernel"
        )

    def state_dynamics(self):
        """
        Block diagonal from (out_dims, x_dims, x_dims) to (state_dims, state_dims)
        """
        F, L, Qc = self._state_dynamics()  # vmap over output dims
        return bdiag(F), bdiag(L), bdiag(Qc)

    def state_transition(self, dt):
        """
        Block diagonal from (out_dims, x_dims, x_dims) to (state_dims, state_dims)

        :param jnp.ndarray dt: time intervals of shape broadcastable with (out_dims,)
        """
        dt = jnp.broadcast_to(dt, (self.out_dims,))
        A = self._state_transition(dt)  # vmap over output dims
        return bdiag(A)

    def state_output(self):
        """
        Pinf is the solution of the Lyapunov equation F P + P F^T + L Qc L^T = 0
        Pinf = solve_continuous_lyapunov(F, Q)

        Block diagonal from (out_dims, x_dims, x_dims) to (state_dims, state_dims)
        """
        H, Pinf = self._state_output()  # vmap over output dims
        return bdiag(H), bdiag(Pinf)

    def _get_LDS(self, dt, timesteps):
        """
        :param jnp.ndarray t: time points of shape broadcastable with (out_dims, timelocs)
        :param jnp.ndarray dt: time points of shape (out_dims, dt_locs)
        """
        H, Pinf = self._state_output()  # (out_dims, sd, sd)
        out_dims, state_dims_per_out = Pinf.shape[0], Pinf.shape[1]
        
        dt = jnp.broadcast_to(dt, (out_dims, dt.shape[-1]))
        Id = jnp.broadcast_to(jnp.eye(state_dims_per_out, dtype=self.array_type), Pinf.shape)
        Zs = jnp.zeros_like(Pinf)
        
        # insert convenience boundaries for kalman filter and smoother
        if dt.shape[1] == 1:  # single dt value, e.g. regular time grid
            A = self._state_transition(dt[:, 0])  # (out_dims, sd, sd)
            Q = vmap(LTI_process_noise, (0, 0), 0)(A, Pinf)
            As = jnp.stack([Id] + [A] * (timesteps - 1) + [Id], axis=0)
            Qs = jnp.stack([Zs] + [Q] * (timesteps - 1) + [Zs], axis=0)
        else:
            A = vmap(self._state_transition, 1, 0)(dt)  # (ts, out_dims, sd, sd)
            Q = vmap(vmap(LTI_process_noise, (0, 0), 0), (0, None), 0)(A, Pinf)
            As = jnp.concatenate((Id[None, ...], A, Id[None, ...]), axis=0)
            Qs = jnp.concatenate((Zs[None, ...], Q, Zs[None, ...]), axis=0)

        return H, Pinf, As, Qs  # (ts, out_dims, sd, sd)

    def get_LDS(self, dt: jnp.ndarray, timesteps: int):
        """
        Block diagonal state form to move out_dims to state dimensions

        :param jnp.ndarray t: time points of shape broadcastable with (out_dims, timelocs)
        :return:
            matrices of shape (ts, sd, sd)
        """
        H, Pinf, As, Qs = self._get_LDS(dt, timesteps)
        
        H, Pinf = bdiag(H), bdiag(Pinf)

        vbdiag = vmap(bdiag)  # vmap over timesteps
        As, Qs = vbdiag(As), vbdiag(Qs)
        return H, Pinf, As, Qs

    ### representations ###
    def spectral_representation(self, omega, return_full = False):
        """
        Calculation of the spectral representation of the kernel, from its state space parameters.
        We return HS(\omega)H^T which is a scalar (directly in observation space.

        Note that in the LEG case this expression simplifies because of the structue of the process
        (it assumes F = NN^T + R - R^T where N is the noise matrix)
        
        :param jnp.array omega: angular frequency float
        """
        F, L, Qc = self.state_dynamics()
        H, _ = self.state_output()
        imag_id = 1j * jnp.eye(self.state_dims)
        
        tmp = F + imag_id * omega
        tmp_inv = jnp.linalg.inv(tmp)
        conj_tmp_inv = jnp.linalg.inv(F - imag_id * omega)
        Komega = tmp_inv @ L @ Qc @ L.T @ conj_tmp_inv.T
        
        if return_full:
            return Komega
        return H @ Komega @ H.T

    def temporal_representation(self, tau, return_full = False):
        """
        Calculation of the temporal representation of the kernel, from its state space parameters
        Computed with H P_inf * expm(F * \tau)^T H^T
        
        :param jnp.array tau: time interval float
        """
        F, _, _ = self.state_dynamics()
        H, Pinf = self.state_output()
        
        A = expm(F * tau)
        At = expm(-F.T * tau)
        P = (A @ Pinf)
        P_ = (Pinf @ At)
        
        Kt = lax.cond(tau > 0.0, lambda: P, lambda:  P_)
        
        if return_full:
            return Kt
        return H @ Kt @ H.T


### kernel classes ###
class WhiteNoise(MarkovianKernel):
    """
    Independent in time, no dynamics
    """

    Qc: jnp.ndarray

    def __init__(self, Qc, array_type=jnp.float32):
        in_dims = 1
        out_dims = Qc.shape[0]
        state_dims = out_dims
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.Qc = self._to_jax(Qc)

    @eqx.filter_vmap()
    def _state_dynamics(self):
        F = []
        L = []
        Pinf = self.Qc
        return F, L, Qc

    @eqx.filter_vmap()
    def _state_output(self):
        H = jnp.ones(self.out_dims)
        Pinf = self.Qc
        return H, Pinf

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        """
        return jnp.zeros((self.state_dims, self.state_dims))
    
    
class WienerProcess(Kernel):
    """
    Wiener process kernels
    """
    def __init__(self, Qc, array_type=jnp.float32):
        in_dims = 1
        out_dims = Qc.shape[0]
        state_dims = out_dims
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.Qc = self._to_jax(Qc)
        
    @eqx.filter_vmap()
    def _state_dynamics(self):
        F = []
        L = []
        return F, L, Qc

    @eqx.filter_vmap()
    def _state_output(self):
        H = self._to_jax([[1.]])
        return H, None

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        """
        return jnp.zeros((self.state_dims, self.state_dims)), self.Qc



class WienerVelocity(Kernel):
    def __init__(self, Qc, array_type=jnp.float32):
        super().__init__(Qc, array_type)
        
        
        
class WienerAcceleration(Kernel):
    def __init__(self, Qc, array_type=jnp.float32):
        super().__init__(Qc, array_type)
    
    
    
    
class DotProduct(Kernel):
    r"""
    Base class for kernels which are functions of :math:`x \cdot z`.
    """
    def __init__(self, in_dims, out_dims, array_type=jnp.float32):
        super().__init__(in_dims, out_dims, None, array_type)

    def _dot_product(self, X, Y, diagonal):
        r"""
        Returns :math:`X \cdot Z`.
        """
        if Y is None:  # autocovariance
            if diagonal:
                return self._K_r(jnp.zeros((*X.shape[:2], 1)))
            Y = X
        
        return X.matmul(Z.permute(0, 1, 3, 2))
    
    
    
class Linear(DotProduct):
    r"""
    Implementation of Linear kernel:
        :math:`k(x, z) = \sigma^2 x \cdot z.`
    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.
    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Constant` kernel.
    """

    def __init__(self, in_dims, array_type=jnp.float32):
        super().__init__(in_dims, array_type)

    def K(self, X, Y, diagonal):
        return self._dot_product(X, Y, diagonal)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:
        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`
    """
    
    degree: int

    def __init__(self, in_dims, bias, degree=1, array_type=jnp.float32):
        """
        :param jnp.ndarray bias: Bias parameter of this kernel. Should be positive.
        :param int degree: Degree :math:`d` of the polynomial.
        """
        super().__init__(in_dims, array_type)
        self.log_bias = jnp.log(self._to_jax(bias)) # N

        if degree < 1:
            raise ValueError("Degree for Polynomial kernel should be a positive integer.")
        self.degree = degree

    def K(self, X, Y, diagonal):
        bias = jnp.exp(self.log_bias)
        
        if diagonal:
            bias = bias[..., 0]
            
        return ((bias + self._dot_product(X, Z, diag)) ** self.degree)



class Cosine(MarkovianKernel):
    """
    Cosine kernel
    """

    pre_omega: jnp.ndarray

    def __init__(self, frequency, array_type=jnp.float32):
        """
        :param jnp.ndarray frequency: radial frequency ω
        """
        in_dims = 1
        out_dims = 1
        state_dims = 2
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.pre_omega = self._to_jax(pre_omega)

    @property
    def omega(self):
        return softplus(self.pre_omega)

    @eqx.filter_vmap()
    def _state_dynamics(self):
        """
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
        omega = softplus(self.pre_omega)
        F = self._to_jax([[0.0, -omega], [omega, 0.0]])
        L = []
        Qc = []
        return F, L, Qc

    @eqx.filter_vmap()
    def _state_output(self):
        H = self._to_jax([1.0, 0.0])
        Pinf = jnp.eye(H.shape[1])
        return H, Pinf

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt)

        :param float dt: step size(s), Δt = tₙ - tₙ₋₁
        :return:
            state transition matrix A (2, 2)
        """
        omega = softplus(self.pre_omega)
        return rotation_matrix(dt, freq)


class LEG(MarkovianKernel):
    """
    State-space formulation of the LEG model. The SDE takes the form
    dz = -G/2zdt + Ndw where G = NN^T + R - R^T
    """

    N: jnp.ndarray
    R: jnp.ndarray
    H: jnp.ndarray
    Lam: jnp.ndarray

    def __init__(self, N, R, H, Lam, array_type=jnp.float32):
        out_dims = H.shape[0]
        state_dims = N.shape[0]
        in_dims = 1
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.N = self._to_jax(N)
        self.R = self._to_jax(R)
        self.H = self._to_jax(H)
        self.Lam = self._to_jax(Lam)

    def parameterize(self):
        # symmetric part
        Q = jnp.diag(softplus(self.N) + 1e-5)  # Q = N@N.T
        # antisymmetric part
        G = Q + (self.R - self.R.T)
        return G, Q

    @eqx.filter_vmap()
    def _state_dynamics(self):
        G, Q = self.parameterize()
        F = -G / 2.0
        L = []
        return F, L, Q

    @eqx.filter_vmap()
    def _state_output(self):
        """
        In this parameterization Pinf is just the identity
        """
        Pinf = jnp.eye(self.state_dims)
        return self.H, Pinf

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        :param float dt: step size(s), Δtₙ = tₙ - tₙ₋₁
        :return:
            state transition matrix A (sd, sd)
        """
        G, _ = self.parameterize()
        return expm(-dt * G / 2)

    def temporal_representation(self, tau):
        """
        Calculation of the temporal representation of the kernel, from its state space parameters
        :return:
            H P_inf*expm(F\tau)^TH^T
        """
        F, Q = self.state_dynamics()
        H, P_inf = self.state_output()
        cond = jnp.sum(tau) >= 0.0

        def pos_tau():
            return H @ P_inf @ expm(tau * F.T) @ H.T

        def neg_tau():
            return H @ P_inf @ expm(jnp.abs(tau) * F.T) @ H.T

        return lax.cond(cond, pos_tau, neg_tau)

    # return H @ P_inf @ expm(tau*F) @ H.T


### lengthscale ###
class DecayingSquaredExponential(Kernel):
    r"""
    Implementation of Decaying Squared Exponential kernel:

        :math:`k(x, z) = \exp\left(-0.5 \times \frac{|x-\beta|^2 + |z-\beta|^2} {l_{\beta}^2}\right) \,
        \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        lengthscale_beta,
        beta,
        array_type=jnp.float32,
    ):
        super().__init__(input_dims, track_dims, f, tensor_type)
        assert (
            self.input_dims == lengthscale.shape[0]
        )  # lengthscale per input dimension

        self.beta = self._to_jax(beta).T  # N, D
        self.log_lengthscale_beta = jnp.log(self._to_jax(lengthscale_beta)).T[
            :, None, None, :
        ]  # N, K, T, D
        self.log_lengthscale = jnp.log(self._to_jax(lengthscale)).T  # N, D

    def K(self, X, Y, diagonal):
        if Y is None:  # autocovariance
            if diagonal:
                return self._K_r(jnp.zeros((*X.shape[:2], 1)))
            Y = X

        scaled_X = X / self.lengthscale  # K, N, T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X**2).sum(-1, keepdim=True)
        Z2 = (scaled_Z**2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
        r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)

        return jnp.exp(
            -0.5
            * (
                r2
                + (((X - self.beta) / self.lengthscale_beta)**2).sum(-1)[..., None]
                + (((Z - self.beta) / self.lengthscale_beta)**2).sum(-1)[..., None, :]
            )
        )

        l = softplus(self.pre_len)
        r_in = self.distance_metric(X, Y, l, diagonal)
        K = self._K_r(r_in)  # (out_dims, pts, pts) or (out_dims, pts, 1)
        return jnp.broadcast_to(K, (self.out_dims, *K.shape[1:]))


class Lengthscale(MarkovianKernel):
    """
    Stationary kernels based on lengthscales
    """

    pre_len: jnp.ndarray
    pre_var: jnp.ndarray

    distance_metric: Callable

    def __init__(
        self, out_dims, state_dims, variance, lengthscale, distance_metric, array_type
    ):
        """
        :param jnp.ndarray variance: σ² (out_dims,)
        :param jnp.ndarray lengthscale: l (out_dims, in_dims)
        """
        in_dims = lengthscale.shape[-1]
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.pre_len = softplus_inv(self._to_jax(lengthscale))
        self.pre_var = softplus_inv(self._to_jax(variance))

        self.distance_metric = distance_metric

    @property
    def variance(self):
        return softplus(self.pre_var)

    @property
    def lengthscale(self):
        return softplus(self.pre_len)

    # kernel
    def K(self, X, Y, diagonal):
        """
        :param jnp.ndarray X: first input (out_dims, num_points, in_dims)
        :param jnp.ndarray Y: second input (out_dims, num_points, in_dims)
        """
        if Y is None:  # autocovariance
            if diagonal:
                return self._K_r(jnp.zeros((*X.shape[:2], 1)))
            Y = X

        l = softplus(self.pre_len)
        r_in = self.distance_metric(X, Y, l, diagonal)
        K = self._K_r(r_in)  # (out_dims, pts, pts) or (out_dims, pts, 1)
        return jnp.broadcast_to(K, (self.out_dims, *K.shape[1:]))

    def _K_r(self, r):
        raise NotImplementedError("kernel function not implemented")

    def _K_omega(self, omega):
        raise NotImplementedError("kernel spectrum not implemented")


        
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

    def __init__(self, out_dims, variance, lengthscale, array_type=jnp.float32):
        super().__init__(
            out_dims, None, variance, lengthscale, _scaled_dist_squared_Rn, array_type
        )

    def _K_r(self, r2):
        variance = softplus(self.pre_var)[:, None, None]
        return variance * jnp.exp(-0.5 * r2)

    def _K_omega(self, omega):
        lengthscale = softplus(self.pre_len)
        variance = softplus(self.pre_var)
        return (
            variance
            * _sqrt_twopi
            * lengthscale
            * jnp.exp(-2 * omega**2 * lengthscale**2)
        )

    def sample_spectrum(self, prng_state, num_samps, RFF_num_feats):
        lengthscale = softplus(self.pre_len)  # (out, d_x)
        variance = softplus(self.pre_var)  # (out_dims,)

        k_std = 1.0 / lengthscale
        ks = k_std[None, :, None, :] * jr.normal(
            prng_state, shape=(num_samps, self.out_dims, RFF_num_feats, self.in_dims)
        )
        amplitude = jnp.sqrt(variance)
        return ks, amplitude

    

class RationalQuadratic(Lengthscale):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param jnp.array scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """

    def __init__(
        self, out_dims, variance, lengthscale, scale_mixture, array_type=jnp.float32
    ):
        super().__init__(
            out_dims, None, variance, lengthscale, _scaled_dist_squared_Rn, array_type
        )
        self.pre_scale_mixture = softplus_inv(self._to_jax(scale_mixture))

    @property
    def scale_mixture(self):
        return softplus(self.pre_scale_mixture)[None, :, None]  # K, N, T

    #     @scale_mixture.setter
    #     def scale_mixture(self):
    #         self._scale_mixture.data = self.lf_inv(scale_mixture)

    def _K_r(self, r2):
        variance = softplus(self.pre_var)[:, None, None]
        scale_mixture = softplus(self.pre_scale_mixture)[:, None, None]
        return variance * (1 + (0.5 / scale_mixture) * r2).pow(-scale_mixture)
    
    

class Matern12(Lengthscale):
    """
    Exponential, i.e. Matern-1/2
    Functions drawn from a GP with this kernel are not differentiable anywhere. 
    """

    def __init__(self, out_dims, variance, lengthscale, array_type=jnp.float32):
        state_dims = out_dims
        super().__init__(
            out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn, array_type
        )

    def _K_r(self, r):
        """
        The kernel equation is
        k(r) = σ² exp{-r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
        σ² is the variance parameter
        """
        variance = softplus(self.pre_var)[:, None, None]
        return variance * jnp.exp(-r)

    @eqx.filter_vmap
    def _state_dynamics(self):
        """
        The associated continuous-time state space model matrices are:
        F      = -1/l
        L      = 1
        Qc     = 2σ²/l
        H      = 1
        Pinf   = σ²
        """
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first  dimension

        F = -1.0 / ell[None, None]
        L = jnp.ones((1, 1), dtype=self.array_type)
        Qc = 2.0 * (var / ell)[None, None]
        return F, L, Qc

    @eqx.filter_vmap(kwargs=dict(dt=0))  # vmap over out_dims
    def _state_transition(self, dt):
        """
        :param float dt: step size(s), Δtₙ = tₙ - tₙ₋₁
        :return:
            state transition matrix A (1, 1)
        """
        ell = softplus(self.pre_len[0])  # first dimension
        A = jnp.exp(-dt / ell)[None, None]
        #Q = LTI_process_noise(A, Pinf)
        return A

    @eqx.filter_vmap
    def _state_output(self):
        var = softplus(self.pre_var)
        H = jnp.ones((1, 1), dtype=self.array_type)  # observation projection
        Pinf = var[None, None]
        return H, Pinf
    
    def sample_spectrum(self, prng_state, num_samps, RFF_num_feats):
        lengthscale = softplus(self.pre_len)  # (out, d_x)
        variance = softplus(self.pre_var)  # (out_dims,)

        k_std = 1.0 / lengthscale
        ks = k_std[None, :, None, :] * jr.t(
            prng_state, df=1., shape=(num_samps, self.out_dims, RFF_num_feats, self.in_dims)
        )
        amplitude = jnp.sqrt(variance)
        return ks, amplitude
    


class Matern32(Lengthscale):
    """
    Matern-3/2 kernel
    Functions drawn from a GP with this kernel are once differentiable
    """

    def __init__(self, out_dims, variance, lengthscale, array_type=jnp.float32):
        
        state_dims = 2 * out_dims
        super().__init__(
            out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn, array_type
        )

    def _K_r(self, r):
        """
        The kernel equation is
        k(r) = σ² (1 + √3r) exp{-√3 r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
        σ² is the variance parameter.
        """
        variance = softplus(self.pre_var)[:, None, None]
        sqrt3 = jnp.sqrt(3.0)
        return variance * (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r)

    @eqx.filter_vmap
    def _state_dynamics(self):
        """
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
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first dimension

        lam = 3.0**0.5 / ell
        F = self._to_jax([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L = self._to_jax([[0], [1]])
        Qc = self._to_jax([[12.0 * 3.0**0.5 / ell**3.0 * var]])
        return F, L, Qc

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁
        :return:
            state transition matrix A (2, 2)
        """
        ell = softplus(self.pre_len[0])  # first dimension

        lam = jnp.sqrt(3.0) / ell
        A = jnp.exp(-dt * lam) * (
            dt * self._to_jax([[lam, 1.0], [-(lam**2.0), -lam]]) + jnp.eye(2)
        )
        return A

    @eqx.filter_vmap
    def _state_output(self):
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first dimension

        H = self._to_jax([[1.0, 0.0]])  # observation projection
        Pinf = self._to_jax([[var, 0.0], [0.0, 3.0 * var / ell**2.0]])
        return H, Pinf
    
    def sample_spectrum(self, prng_state, num_samps, RFF_num_feats):
        lengthscale = softplus(self.pre_len)  # (out, d_x)
        variance = softplus(self.pre_var)  # (out_dims,)

        k_std = 1.0 / lengthscale
        ks = k_std[None, :, None, :] * jr.t(
            prng_state, df=2., shape=(num_samps, self.out_dims, RFF_num_feats, self.in_dims)
        )
        amplitude = jnp.sqrt(variance)
        return ks, amplitude
    


class Matern52(Lengthscale):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice differentiable
    """

    def __init__(self, out_dims, variance, lengthscale, array_type=jnp.float32):
        state_dims = 3 * out_dims
        super().__init__(
            out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn, array_type
        )

    def _K_r(self, r):
        """
        The kernel equation is
        k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
        σ² is the variance parameter.
        """
        variance = softplus(self.pre_var)[:, None, None]
        sqrt5 = jnp.sqrt(5.0)
        return variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * r**2) * jnp.exp(-sqrt5 * r)

    @eqx.filter_vmap
    def _state_dynamics(self):
        """
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
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first dimension

        lam = 5.0**0.5 / ell
        F = self._to_jax(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3.0), -3.0 * lam**2.0, -3.0 * lam],
            ]
        )
        L = self._to_jax([[0.0], [0.0], [1.0]])
        Qc = self._to_jax([[var * 400.0 * 5.0**0.5 / 3.0 / ell**5.0]])
        return F, L, Qc

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        :param float dt: step size(s), Δtₙ = tₙ - tₙ₋₁
        :return:
            state transition matrix A (3, 3)
        """
        ell = softplus(self.pre_len[0])  # first dimension

        lam = jnp.sqrt(5.0) / ell
        dtlam = dt * lam
        A = jnp.exp(-dtlam) * (
            dt
            * self._to_jax(
                [
                    [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                    [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                    [
                        lam**3 * (0.5 * dtlam - 1.0),
                        lam**2 * (dtlam - 3),
                        lam * (0.5 * dtlam - 2.0),
                    ],
                ]
            )
            + jnp.eye(3)
        )
        return A

    @eqx.filter_vmap
    def _state_output(self):
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first dimension

        H = self._to_jax([[1.0, 0.0, 0.0]])  # observation projection
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf = self._to_jax(
            [
                [var, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * var / ell**4.0],
            ]
        )
        return H, Pinf
    
    def sample_spectrum(self, prng_state, num_samps, RFF_num_feats):
        lengthscale = softplus(self.pre_len)  # (out, d_x)
        variance = softplus(self.pre_var)  # (out_dims,)

        k_std = 1.0 / lengthscale
        ks = k_std[None, :, None, :] * jr.t(
            prng_state, df=3., shape=(num_samps, self.out_dims, RFF_num_feats, self.in_dims)
        )
        amplitude = jnp.sqrt(variance)
        return ks, amplitude
    
    


class Matern72(Lengthscale):
    """
    Matern-7/2 kernel
    """

    def __init__(self, out_dims, variance, lengthscale, array_type=jnp.float32):
        state_dims = 4 * out_dims
        super().__init__(
            out_dims, state_dims, variance, lengthscale, _scaled_dist_Rn, array_type
        )

    # kernel
    def _K_r(self, r):
        variance = softplus(self.pre_var)[:, None, None]
        sqrt7 = jnp.sqrt(7.0)
        return (
            variance
            * (
                1.0
                + sqrt7 * r
                + 14.0 / 5.0 * r**2
                + 7.0 * sqrt7 / 15.0 * r**3
            )
            * jnp.exp(-sqrt7 * r)
        )

    # state space
    @eqx.filter_vmap
    def _state_dynamics(self):
        """
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
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first dimension

        lam = 7.0**0.5 / ell
        F = self._to_jax(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-(lam**4.0), -4.0 * lam**3.0, -6.0 * lam**2.0, -4.0 * lam],
            ]
        )
        L = self._to_jax([[0.0], [0.0], [0.0], [1.0]])
        Qc = self._to_jax([[var * 10976.0 * 7.0**0.5 / 5.0 / ell**7.0]])

        return F, L, Qc

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        ell = softplus(self.pre_len[0])  # first dimension

        lam = jnp.sqrt(7.0) / ell
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam**2
        A = jnp.exp(-dtlam) * (
            dt
            * self._to_jax(
                [
                    [
                        lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),
                        1.0 + dtlam + 0.5 * dtlam2,
                        0.5 * dt * (1.0 + dtlam),
                        dt**2 / 6,
                    ],
                    [
                        -dtlam2 * lam**2.0 / 6.0,
                        lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                        1.0 + dtlam - 0.5 * dtlam2,
                        dt * (0.5 - dtlam / 6.0),
                    ],
                    [
                        lam3 * dtlam * (dtlam / 6.0 - 0.5),
                        dtlam * lam2 * (0.5 * dtlam - 2.0),
                        lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),
                        1.0 - dtlam + dtlam2 / 6.0,
                    ],
                    [
                        lam2**2 * (dtlam - 1.0 - dtlam2 / 6.0),
                        lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                        lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),
                        lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0),
                    ],
                ]
            )
            + jnp.eye(4)
        )
        return A

    @eqx.filter_vmap
    def _state_output(self):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt)
        
        :param float dt: step size(s), Δtₙ = tₙ - tₙ₋₁
        :return:
            state transition matrix A (4, 4)
        """
        var = softplus(self.pre_var)
        ell = softplus(self.pre_len[0])  # first dimension
        
        H = self._to_jax([[1.0, 0.0, 0.0, 0.0]])  # observation projection
        kappa = 7.0 / 5.0 * var / ell**2.0
        kappa2 = 9.8 * var / ell**4.0

        Pinf = self._to_jax(
            [
                [var, 0.0, -kappa, 0.0],
                [0.0, kappa, 0.0, -kappa2],
                [-kappa, 0.0, kappa2, 0.0],
                [0.0, -kappa2, 0.0, 343.0 * var / ell**6.0],
            ]
        )
        return H, Pinf
    
    def sample_spectrum(self, prng_state, num_samps, RFF_num_feats):
        lengthscale = softplus(self.pre_len)  # (out, d_x)
        variance = softplus(self.pre_var)  # (out_dims,)

        k_std = 1.0 / lengthscale
        ks = k_std[None, :, None, :] * jr.t(
            prng_state, df=4., shape=(num_samps, self.out_dims, RFF_num_feats, self.in_dims)
        )
        amplitude = jnp.sqrt(variance)
        return ks, amplitude
    
    


### combinations ###
class MarkovSparseKronecker(MarkovianKernel):
    """
    Kronecker product of a Markovian kernel with a sparse kernel
    """

    markov_factor: MarkovianKernel
    sparse_factor: Kernel

    induc_locs: jnp.ndarray

    def __init__(self, markov_factor, sparse_factor, induc_locs):
        """
        :param jnp.ndarray induc_locs: inducing locations of shape (out_dims, spatial_locs, x_dims)
        """
        assert markov_factor.array_type == sparse_factor.array_type
        in_dims = 1 + sparse_factor.in_dims
        assert markov_factor.out_dims == sparse_factor.out_dims
        num_induc = induc_locs.shape[0]
        state_dims = markov_factor.state_dims * num_induc
        super().__init__(
            in_dims, markov_factor.out_dims, state_dims, markov_factor.array_type
        )
        self.markov_factor = markov_factor
        self.sparse_factor = sparse_factor
        self.induc_locs = self._to_jax(induc_locs)  # (out_dims, num_induc, x_dims)

    def sparse_conditional(self, x_eval, jitter):
        """
        The spatial conditional covariance and posterior mean on whitened u(t)

        :param jnp.ndarray x_eval: evaluation locations shape (out_dims, ts, x_dims)
        """
        num_induc = self.induc_locs.shape[1]

        Kxx = self.sparse_factor.K(x_eval, None, True)  # (out_dims, ts, 1)
        Kzz = self.sparse_factor.K(self.induc_locs, None, False)

        eps_I = jitter * jnp.eye(num_induc)
        chol_Kzz = cholesky(Kzz + eps_I)

        Kzx = self.sparse_factor.K(
            self.induc_locs, x_eval, False
        )  # (outdims, num_induc, ts)
        Linv_Kzx = solve_triangular(chol_Kzz, Kzx, lower=True)

        C_krr = Linv_Kzx.transpose(0, 2, 1)
        C_nystrom = Kxx - (C_krr**2).sum(-1, keepdims=True)  # (out_dims, ts, 1)
        return C_krr, C_nystrom

    def state_transition(self, dt):
        """
        :return:
            state transition matrices with Kronecker factorization
        """
        num_induc = self.induc_locs.shape[1]
        A = self.markov_factor._state_transition(
            dt
        )  # (out_dims, state_dims, state_dims)
        return id_kronecker(
            num_induc, A
        )  # (out_dims, num_induc*state_dims, num_induc*state_dims)

    def _get_LDS(self, dt, timesteps):
        return self.markov_factor._get_LDS(dt, timesteps)

    def get_LDS(self, dt, timesteps):
        """
        :param jnp.ndarray dt: time intervals of shape (out_dims, dt_locs)
        """
        num_induc = self.induc_locs.shape[1]
        H, Pinf, As, Qs = self._get_LDS(dt, timesteps)

        # kronecker structure
        H, Pinf = id_kronecker(num_induc, H), id_kronecker(num_induc, Pinf)
        As, Qs = id_kronecker(num_induc, As), id_kronecker(num_induc, Qs)
        return H, Pinf, As, Qs


class Product(Kernel):
    """
    The product kernel for handling multi-dimensional input
    """

    kernels: List[Kernel]
    dims_list: List[List[int]]

    def __init__(self, kernels, dims_list):
        """
        :param list dims_list: a list of dimension indices list per kernel that are used
        """
        assert len(dims_list) == len(kernels)
        in_dims = max([max(dl) for dl in dims_list]) + 1
        out_dims = kernels[0].out_dims
        for k in kernels[1:]:
            assert k.out_dims == out_dims
        super().__init__(in_dims, out_dims)
        self.kernels = kernels
        self.dims_list = dims_list

    # kernel
    def K(self, X, Y, diagonal):
        """
        :param jnp.ndarray X: input of shape (num_points, out_dims, dims)
        """
        K = 1.0
        for en, k in enumerate(self.kernels):
            inds = self.dims_list[en]
            K = K * k.K(X[..., inds], None if Y is None else Y[..., inds], diagonal)

        return K


class Sum(Kernel):
    """
    Sum kernels
    """

    kernels: List[Kernel]

    def __init__(self, kernels):
        """
        :param list dims_list: a list of dimension indices list per kernel that are used
        """
        assert len(dims_list) == len(kernels)
        super().__init__(in_dims, out_dims)
        self.kernels = kernels
        self.dims_list = dims_list

    # kernel
    def K(self, X, Y, diagonal):
        """
        :param jnp.ndarray X: input of shape (num_points, out_dims, dims)
        """
        K = 1.0
        for en, k in enumerate(self.kernels):
            inds = self.dims_list[en]
            K = K * k.K(X[..., inds], Y[..., inds], diagonal)

        return K


class GroupMarkovian(MarkovianKernel):
    """
    A group of LDSs with the same state dimension
    """
    
    kernels: List[MarkovianKernel]

    def __init__(self, kernels):
        out_dims = kernels[0].out_dims
        state_dims = kernels[0].state_dims
        in_dims = kernels[0].in_dims
        array_type = kernels[0].array_type
        
        for k in kernels[1:]:
            if out_dims != k.out_dims:
                raise ValueError("Output dimensions must be the same for all kernels")
            if state_dims != k.state_dims:
                raise ValueError("State dimensions must be the same for all kernels")
            if in_dims != k.in_dims:
                raise ValueError("Input dimensions must be the same for all kernels")
            assert array_type == k.array_type
            
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.kernels = kernels

    def _get_LDS(self, dt, timesteps):
        """
        :param jnp.ndarray t: time points of shape broadcastable with (out_dims, timelocs)
        :param jnp.ndarray dt: time points of shape (out_dims, dt_locs)
        :return:
            matrices of shape (orders, ts, out_dims, sd, sd)
        """
        H, Pinf, As, Qs = [], [], [], []
        for k in self.kernels:
            H_, Pinf_, As_, Qs_ = k._get_LDS(dt, timesteps)
            H.append(H_)
            Pinf.append(Pinf_)
            As.append(As_)
            Qs.append(Qs_)
            
        return [jnp.stack(A, axis=0) for A in [H, Pinf, As, Qs]]

    def get_LDS(self, dt: jnp.ndarray, timesteps: int):
        """
        Block diagonal state form to move out_dims to state dimensions

        :param jnp.ndarray t: time points of shape broadcastable with (out_dims, timelocs)
        :return:
            matrices of shape (orders, ts, sd, sd)
        """
        vbdiag = vmap(bdiag)  # vmap over orders
        vvbdiag = vmap(vbdiag)  # vmap over timesteps
        
        H, Pinf, As, Qs = self._get_LDS(dt, timesteps)
        H, Pinf = vbdiag(H), vbdiag(Pinf)
        As, Qs = vvbdiag(As), vvbdiag(Qs)
        return H, Pinf, As, Qs
    
    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs

        :param jnp.ndarray dt: step size(s), Δt = tₙ - tₙ₋₁ (out_dims, timelocs)
        :return:
            state transition matrix A (orders, sd, sd)
        """
        A = []
        for k in self.kernels:
            A.append(k.state_transition(dt))
            
        return jnp.stack(A, axis=0)


class StackMarkovian(MarkovianKernel):
    """
    A stack of independent GP LDSs
    This class stacks the state space models, each process increasing the readout dimensions
    """

    kernels: List[MarkovianKernel]

    def __init__(self, kernels):
        out_dims = kernels[0].out_dims
        state_dims = kernels[0].state_dims
        in_dims = kernels[0].in_dims
        array_type = kernels[0].array_type
        
        for k in kernels[1:]:
            out_dims += k.state_dims
            state_dims += k.state_dims
            if in_dims != k.in_dims:
                raise ValueError("Input dimensions must be the same for all kernels")
            assert array_type == k.array_type

        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.kernels = kernels

    # kernel
    def K(self, X, Y, diagonal):
        for i in range(1, len(self.kernels)):
            r_in = self.distance_metric(X, Y, diagonal)
            return self.K_r(r_in)

    # state space
    def state_dynamics(self):
        F, L, Qc = self.kernels[0].state_dynamics()

        for i in range(1, len(self.kernels)):
            F_, L_, Qc_ = self.kernels[i].state_dynamics()
            F = jnp.block(
                [
                    [F, jnp.zeros([F.shape[0], F_.shape[1]])],
                    [jnp.zeros([F_.shape[0], F.shape[1]]), F_],
                ]
            )
            L = jnp.block(
                [
                    [L, jnp.zeros([L.shape[0], L_.shape[1]])],
                    [jnp.zeros([L_.shape[0], L.shape[1]]), L_],
                ]
            )
            Qc = jnp.block(
                [
                    [Qc, jnp.zeros([Qc.shape[0], Qc_.shape[1]])],
                    [jnp.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_],
                ]
            )
        return F, L, Qc

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs

        :param jnp.ndarray dt: step size(s), Δt = tₙ - tₙ₋₁
        :return:
            state transition matrix A (sd, sd)
        """
        A = self.kernels[0].state_transition(dt)

        for i in range(1, len(self.kernels)):
            A_ = self.kernels[i].state_transition(dt)
            A = jnp.block(
                [
                    [A, jnp.zeros([A.shape[0], A_.shape[0]])],
                    [jnp.zeros([A_.shape[0], A.shape[0]]), A_],
                ]
            )
        return A

    def state_output(self):
        H, Pinf = self.kernels[0].state_output()

        for i in range(1, len(self.kernels)):
            H_, Pinf_ = self.kernels[i].state_output()
            H = jnp.block(
                [
                    [H, jnp.zeros([H.shape[0], H_.shape[1]])],
                    [jnp.zeros([H_.shape[0], H.shape[1]]), H_],
                ]
            )
            Pinf = jnp.block(
                [
                    [Pinf, jnp.zeros([Pinf.shape[0], Pinf_.shape[1]])],
                    [jnp.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_],
                ]
            )

        return H, Pinf


class SumMarkovian(MarkovianKernel):
    """
    A sum of GP priors.
    This class stacks the state space models to produce their sum.
    This class differs from Independent only in the measurement model.
    """

    kernels: List[MarkovianKernel]

    def __init__(self, kernels):
        out_dims = kernels[0].out_dims
        state_dims = kernels[0].state_dims
        in_dims = kernels[0].in_dims
        array_type = kernels[0].array_type
        
        for k in kernels[1:]:
            if out_dims != k.out_dims:
                raise ValueError("Output dimensions must be the same for all kernels")
            state_dims += k.state_dims
            if in_dims != k.in_dims:
                raise ValueError("Input dimensions must be the same for all kernels")
            assert array_type == k.array_type
            
        super().__init__(in_dims, out_dims, state_dims, array_type)
        self.kernels = kernels

    def state_dynamics(self):
        F, L, Qc = self.kernels[0].state_dynamics()
        # H, Pinf =

        for i in range(1, len(self.kernels)):
            F_, L_, Qc_, H_, Pinf_ = self.kernels[i].state_dynamics()
            F = jnp.block(
                [
                    [F, jnp.zeros([F.shape[0], F_.shape[1]])],
                    [jnp.zeros([F_.shape[0], F.shape[1]]), F_],
                ]
            )
            L = jnp.block(
                [
                    [L, jnp.zeros([L.shape[0], L_.shape[1]])],
                    [jnp.zeros([L_.shape[0], L.shape[1]]), L_],
                ]
            )
            Qc = jnp.block(
                [
                    [Qc, jnp.zeros([Qc.shape[0], Qc_.shape[1]])],
                    [jnp.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_],
                ]
            )
            H = jnp.block([H, H_])
            Pinf = jnp.block(
                [
                    [Pinf, jnp.zeros([Pinf.shape[0], Pinf_.shape[1]])],
                    [jnp.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_],
                ]
            )
        return F, L, Qc, H, Pinf

    def measurement_model(self, r=None):
        H = self.kernels[0].measurement_model(r)
        for i in range(1, len(self.kernels)):
            H_ = self.kernels[i].measurement_model(r)
            H = jnp.block([H, H_])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs
        :param jnp.ndarray dt: step size(s), Δt = tₙ - tₙ₋₁
        :return:
            state transition matrix A (sd, sd)
        """
        A = self.kernels[0].state_transition(dt)
        for i in range(1, len(self.kernels)):
            A_ = self.kernels[i].state_transition(dt)
            A = jnp.block(
                [
                    [A, jnp.zeros([A.shape[0], A_.shape[0]])],
                    [jnp.zeros([A_.shape[0], A.shape[0]]), A_],
                ]
            )
        return A

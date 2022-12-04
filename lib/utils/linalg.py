import jax.numpy as jnp
from jax import vmap, random, tree_map

from jax.scipy.special import erfc
from jax.scipy.linalg import cho_factor, cho_solve, expm
from jax.numpy.linalg import cholesky


import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import interp1d

#from sklearn.cross_decomposition import CCA


import itertools




### linear algebra ###
def enforce_positive_diagonal(K, lower_lim=1e-6):
    """
    Check whether matrix K has positive diagonal elements.
    If not, then replace the negative elements with default value 0.01
    """
    K_diag = jnp.diag(jnp.diag(K))
    K = jnp.where(jnp.any(jnp.diag(K) < 0), jnp.where(K_diag < 0, lower_lim, K_diag), K)
    return K


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factoristion
    """
    Lt = cho_factor(P) # upper triangular default
    return cho_solve(Lt, Q)


def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    Lt = cho_factor(P)
    return cho_solve(Lt, jnp.eye(P.shape[0]))




def rotation_matrix(dt, omega):
    """
    Discrete time rotation matrix
    :param dt: step size [1]
    :param omega: frequency [1]
    :return:
        R: rotation matrix [2, 2]
    """
    R = jnp.array([
        [jnp.cos(omega * dt), -jnp.sin(omega * dt)],
        [jnp.sin(omega * dt),  jnp.cos(omega * dt)]
    ])
    return R



def solve_continuous_lyapunov(A, Q):
    """
    Solves the continuous Lyapunov equation A@X + X@A.T + Q = 0, using the vectorized form i.e
    (In  ⊗ A)vec(X) + (A ⊗ In)vec(X) = -vec(Q)
    This returns the opposite of the scipy.linalg.solve_continuous_lyapunov solver but not sure why  (for stable systems, for unstable systems results seem completely disconnected) 
    """
    n = jnp.shape(A)[0]
    In = jnp.eye(n)
    T = jnp.kron(In, A) + jnp.kron(A, In)
    neg_vecQ = -jnp.ravel(Q)
    vecX = jnp.linalg.solve(T, neg_vecQ)
    return jnp.reshape(vecX, (n,n))


#this one seems to match the scipy result
def solve_discrete_lyapunov(A, Q):
    """
    Solves the continuous Lyapunov equation A@X + X@A.T + Q = 0, using the vectorized form i.e
    (I_{n^2}  - A.T⊗A)vec(X)= vec(Q)
    This returns the same as the scipy.linalg.solve_discrete_lyapunov solver (for stable systems, for unstable systems results seem completely disconnected)
    """
    n = jnp.shape(A)[0]
    In2 = jnp.eye(n*n)
    T = In2 - jnp.kron(A.T, A)
    vecQ = jnp.ravel(Q)
    vecX = jnp.linalg.solve(T, vecQ)
    return jnp.reshape(vecX, (n,n))



def eigenvalues(w):
    eigs = np.linalg.eigvals(w)
    re = eigs.real
    im = eigs.imag
    return(np.concatenate([re[:,None],im[:,None]],axis=1))

def jeigenvalues(w):
    eigs = jnp.linalg.eigvals(w)
    re = eigs.real
    im = eigs.imag
    return(jnp.concatenate([re[:,None],im[:,None]],axis=1))



def get_blocks(A, num_blocks, block_size):
    k = 0
    A_ = []
    for _ in range(num_blocks):
        A_.append(A[k:k+block_size, k:k+block_size])
        k += block_size
    return jnp.stack(A_, axis=0)



### LDS ###
def compute_kernel(delta_t, F, Pinf, H):
    """
    delta_t is positive and increasing
    """
    A = vmap(expm)(F[None, ...]*delta_t[:, None, None])
    At = vmap(expm)(-F.T[None, ...]*delta_t[:, None, None])
    P = (A[..., None] * Pinf[None, None, ...]).sum(-2)
    P_ = (Pinf[None, ..., None] * At[:, None, ...]).sum(-2)

    delta_t = np.broadcast_to(delta_t[:, None, None], P.shape)
    Kt = H[None, ...] @ jnp.where(delta_t > 0., P, P_) @ H.T[None, ...]
    return Kt



def discrete_transitions(F, L, Qc):
    """
    """
    A = expm(F*dt)
    Pinf = solve_continuous_lyapunov(F, L @ L.T * Qc)
    Q = Pinf - A @ Pinf @ A.T
    return A, Pinf


### cubature ###
def discretegrid(xy, w, nt):
    """
    Convert spatial observations to a discrete intensity grid
    :param xy: observed spatial locations as a two-column vector
    :param w: observation window, i.e. discrete grid to be mapped to, [xmin xmax ymin ymax]
    :param nt: two-element vector defining number of bins in both directions
    """
    # Make grid
    x = np.linspace(w[0], w[1], nt[0] + 1)
    y = np.linspace(w[2], w[3], nt[1] + 1)
    X, Y = np.meshgrid(x, y)

    # Count points
    N = np.zeros([nt[1], nt[0]])
    for i in range(nt[0]):
        for j in range(nt[1]):
            ind = (xy[:, 0] >= x[i]) & (xy[:, 0] < x[i + 1]) & (xy[:, 1] >= y[j]) & (xy[:, 1] < y[j + 1])
            N[j, i] = np.sum(ind)
    return X[:-1, :-1].T, Y[:-1, :-1].T, N.T



def gauss_hermite(dim, num_quad_pts):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.
    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])
    
    Return weights and sigma-points for Gauss-Hermite cubature
    """
    gh_x, gh_w = hermgauss(num_quad_pts)
    sigma_pts = np.array(list(itertools.product(*(gh_x,) * dim)))  # H**DxD
    weights = np.prod(np.array(list(itertools.product(*(gh_w,) * dim))), 1)  # H**D
    
    sigma_pts = jnp.sqrt(2.) * sigma_pts.T
    weights = weights.T * jnp.pi**(-dim/2.)  # scale weights by 1/√π
    return sigma_pts, weights

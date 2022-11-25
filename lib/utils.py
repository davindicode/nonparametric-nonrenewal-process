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



### pytree ###
def copy_pytree(_pytree):
    """
    None values are not regarded as leaves, so are skipped
    """
    return tree_map(
        lambda x: jnp.array(x), _pytree)



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



### spike trains ###
def bin_data(bin_size, bin_time, spikes, track_samples, 
             behaviour_data=None, average_behav=True, binned=False):
    """
    Bin the spike train into a given bin size.
    
    :param int bin_size: desired binning of original time steps into new bin
    :param float bin_time: time step of each original bin or time point
    :param np.array spikes: input spikes in train or index format
    :param int track_samples: number of time steps in the recording
    :param tuple behaviour_data: input behavioural time series
    :param bool average_behav: takes the middle element in bins for behavioural data if False
    :param bool binned: spikes is a spike train if True (trials, neurons, time), otherwise 
                        it is a list of spike time indices (trials, neurons)*[spike indices]
    :return:
        tbin, resamples, rc_t, rcov_t
    """
    tbin = bin_size*bin_time
    resamples = int(np.floor(track_samples/bin_size))
    centre = bin_size // 2
    # leave out data with full bins
    
    rcov_t = ()
    if behaviour_data is not None:
        if isinstance(average_behav, list) is False:
            average_behav = [average_behav for _ in range(len(behaviour_data))]
        
        for k, cov_t in enumerate(behaviour_data):
            if average_behav[k]:
                rcov_t += (cov_t[:resamples*bin_size].reshape(resamples, bin_size).mean(1),)
            else:
                rcov_t += (cov_t[centre:resamples*bin_size:bin_size],)
            
    if binned:
        rc_t = spikes[:, :resamples*bin_size].reshape(spikes.shape[0], resamples, bin_size).sum(-1)
    else:
        units = len(spikes)
        rc_t = np.zeros((units, resamples))
        for u in range(units):
            retimes = np.floor(spikes[u]/bin_size).astype(int)
            np.add.at(rc_t[u], retimes[retimes < resamples], 1)
        
    return tbin, resamples, rc_t, rcov_t



def binned_to_indices(spiketrain):
    """
    Converts a binned spike train into spike time indices (with duplicates)
    
    :param np.array spiketrain: the spike train to convert
    :returns: spike indices denoting spike times in units of time bins
    :rtype: np.array
    """
    spike_ind = spiketrain.nonzero()[0]
    bigger = np.where(spiketrain > 1)[0]
    add_on = (spike_ind,)
    for b in bigger:
        add_on += (b*np.ones(int(spiketrain[b])-1, dtype=int),)
    spike_ind = np.concatenate(add_on)
    return np.sort(spike_ind)



def covariates_at_spikes(spiketimes, behaviour_data):
    """
    Returns tuple of covariate arrays at spiketimes for all neurons
    """
    cov_s = tuple([] for n in behaviour_data)
    units = len(spiketimes)
    for u in range(units):
        for k, cov_t in enumerate(behaviour_data):
            cov_s[k].append(cov_t[spiketimes[u]])
            
    return cov_s






### functions ###
def expsum(a, axis=0):
    """
    Numerically stabilized
    """
    a_max = a.max(axis=axis, keepdims=True)
    return jnp.exp(a_max.sum(axis=axis)) * jnp.exp((a - a_max).sum(axis=axis))



def softplus_list(x_):
    """
    Softplus positivity mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = [jnp.log(1 + jnp.exp(-jnp.abs(x_[0]))) + jnp.maximum(x_[0], 0)]
    for i in range(1, len(x_)):
        if x_[i] is not []:
            y_ = y_ + [jnp.log(1 + jnp.exp(-jnp.abs(x_[i]))) + jnp.maximum(x_[i], 0)]
    return y_


def softplus_inv_list(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = x_
    for i in range(len(x_)):
        if x_[i] is not []:
            y_[i] = jnp.log(1 - jnp.exp(-jnp.abs(x_[i]))) + jnp.maximum(x_[i], 0)
    return y_


def softplus(x):
    return jnp.log(1 + jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0) # numerically stabilized


def sigmoid(x):
    return jnp.exp(x) / (jnp.exp(x) + 1.)


def softplus_inv(x):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    if x is None:
        return x
    else:
        return jnp.log(1 - jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0) # numerically stabilized

    
def logphi(z):
    """
    Calculate the log Gaussian CDF, used for closed form moment matching when the EP power is 1,
        logΦ(z) = log[(1 + erf(z / √2)) / 2]
    for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    and its derivative w.r.t. z
        dlogΦ(z)/dz = 𝓝(z|0,1) / Φ(z)
    :param z: input value, typically z = (my) / √(1 + v) [scalar]
    :return:
        lp: logΦ(z) [scalar]
        dlp: dlogΦ(z)/dz [scalar]
    """
    z = jnp.real(z)
    # erfc(z) = 1 - erf(z) is the complementary error function
    lp = jnp.log(erfc(-z / jnp.sqrt(2.0)) / 2.0)  # log Φ(z)
    dlp = jnp.exp(-z * z / 2.0 - lp) / jnp.sqrt(2.0 * jnp.pi)  # derivative w.r.t. z
    return lp, dlp




### Monte Carlo ###
def sample_gaussian_noise(key, mean, Lcov):
    gaussian_sample = mean + Lcov @ random.normal(key, shape=mean.shape)
    return gaussian_sample


def mc_sample(dim, key, approx_points):
    z = random.normal(key, shape=(dim, approx_points))
    w = np.ones((approx_points,)) / approx_points
    return z, w


def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95], smooth_length=1, conv_mode='same'):
    """
    Compute quantile intervals from samples, samples has shape (sample_dim, event_dims..., T).
    
    :param torch.tensor samples: input samples of shape (MC, num_points, dims)
    :param list percentiles: list of percentile values to look at
    :param int smooth_length: time steps over which to smooth with uniform block
    :returns: list of tensors representing percentile boundaries
    :rtype: list
    """
    num_samples, T = samples.shape[0], samples.shape[-1]
        
    samples = jnp.sort(samples, axis=0)
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]
    
    if smooth_length > 1: # smooth the samples
        window = 1./smooth_length * jnp.ones(smooth_length)
        percentile_samples = [
            vmap(jnp.convolve, (1, None, None), 1)(percentile_sample, window, conv_mode)
            for percentile_sample in percentile_samples
        ]
    
    return percentile_samples



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

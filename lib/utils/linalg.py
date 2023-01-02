import functools
import itertools

import jax.numpy as jnp

import numpy as np
from jax import random, tree_map, vmap
from jax.numpy import array, concatenate
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_factor, cho_solve, expm
from jax.tree_util import register_pytree_node_class

from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss


### linear algebra ###
def solve_PSD(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factoristion
    """
    Lt = cho_factor(P)  # upper triangular default
    return cho_solve(Lt, Q)


def inv_PSD(P):
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
    R = jnp.array(
        [
            [jnp.cos(omega * dt), -jnp.sin(omega * dt)],
            [jnp.sin(omega * dt), jnp.cos(omega * dt)],
        ]
    )
    return R


def solve_continuous_lyapunov(A, Q):
    """
    Solves the continuous Lyapunov equation A@X + X@A.T + Q = 0, using the vectorized form i.e
    (In  ⊗ A)vec(X) + (A ⊗ In)vec(X) = -vec(Q)
    This returns the opposite of the scipy.linalg.solve_continuous_lyapunov solver
    (for stable systems, for unstable systems results seem completely disconnected)
    """
    n = jnp.shape(A)[0]
    In = jnp.eye(n)
    T = jnp.kron(In, A) + jnp.kron(A, In)
    neg_vecQ = -jnp.ravel(Q)
    vecX = jnp.linalg.solve(T, neg_vecQ)
    return jnp.reshape(vecX, (n, n))


def solve_discrete_lyapunov(A, Q):
    """
    this one seems to match the scipy result

    Solves the continuous Lyapunov equation A@X + X@A.T + Q = 0, using the vectorized form i.e
    (I_{n^2}  - A.T⊗A)vec(X)= vec(Q)
    This returns the same as the scipy.linalg.solve_discrete_lyapunov solver
    (for stable systems, for unstable systems results seem completely disconnected)
    """
    n = jnp.shape(A)[0]
    In2 = jnp.eye(n * n)
    T = In2 - jnp.kron(A.T, A)
    vecQ = jnp.ravel(Q)
    vecX = jnp.linalg.solve(T, vecQ)
    return jnp.reshape(vecX, (n, n))


def get_blocks(A, num_blocks, block_size):
    k = 0
    A_ = []
    for _ in range(num_blocks):
        A_.append(A[k : k + block_size, k : k + block_size])
        k += block_size
    return jnp.stack(A_, axis=0)


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
            ind = (
                (xy[:, 0] >= x[i])
                & (xy[:, 0] < x[i + 1])
                & (xy[:, 1] >= y[j])
                & (xy[:, 1] < y[j + 1])
            )
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

    sigma_pts = np.sqrt(2.0) * sigma_pts
    weights = weights * np.pi ** (-dim / 2.0)  # scale weights by 1/√π
    return sigma_pts, weights


def gauss_legendre(dim, num_quad_pts):
    """
    Gauss-Legendre
    """
    gl_x, gl_w = leggauss(num_quad_pts)
    sigma_pts = np.array(list(itertools.product(*(gl_x,) * dim)))  # H**DxD
    weights = np.prod(np.array(list(itertools.product(*(gl_w,) * dim))), 1)  # H**D
    return sigma_pts, weights


def newton_cotes(f, h, order="trapezoid"):
    """
    Integrate from regularly sampled function points

    :param jnp.ndarray f: function values (num_locs,)
    """
    num_locs = f.shape[0]

    if order == "trapezoid":
        coeff = jnp.ones(num_locs)
        coeff = coeff.at[0].set(0.5).at[-1].set(0.5)
        return h * (coeff * f).sum(-1)

    elif order == "simpson13":
        return

    else:
        raise ValueError("Invalid Newton-Cotes rule")


def cumtrapz(f):
    """
    cumulative trapezoid rule
    assume f is left edges of intervals
    """
    cs = jnp.cumsum(f)
    return 0.5 * cs


@register_pytree_node_class
class InterpolatedUnivariateSpline(object):
    def __init__(self, x, y, k=3, endpoints="not-a-knot", coefficients=None):
        """JAX implementation of kth-order spline interpolation.

        This class aims to reproduce scipy's InterpolatedUnivariateSpline
        functionality using JAX. Not all of the original class's features
        have been implemented yet, notably
        - `w`    : no weights are used in the spline fitting.
        - `bbox` : we assume the boundary to always be [x[0], x[-1]].
        - `ext`  : extrapolation is always active, i.e., `ext` = 0.
        - `k`    : orders `k` > 3 are not available.
        - `check_finite` : no such check is performed.

        (The relevant lines from the original docstring have been included
        in the following.)

        Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
        Spline function passes through all provided points. Equivalent to
        `UnivariateSpline` with s = 0.

        Parameters
        ----------
        x : (N,) array_like
            Input dimension of data points -- must be strictly increasing
        y : (N,) array_like
            input dimension of data points
        k : int, optional
            Degree of the smoothing spline.  Must be 1 <= `k` <= 3.
        endpoints : str, optional, one of {'natural', 'not-a-knot'}
            Endpoint condition for cubic splines, i.e., `k` = 3.
            'natural' endpoints enforce a vanishing second derivative
            of the spline at the two endpoints, while 'not-a-knot'
            ensures that the third derivatives are equal for the two
            left-most `x` of the domain, as well as for the two
            right-most `x`. The original scipy implementation uses
            'not-a-knot'.
        coefficients: list, optional
            Precomputed parameters for spline interpolation. Shouldn't be set
            manually.

        See Also
        --------
        UnivariateSpline : Superclass -- allows knots to be selected by a
            smoothing condition
        LSQUnivariateSpline : spline for which knots are user-selected
        splrep : An older, non object-oriented wrapping of FITPACK
        splev, sproot, splint, spalde
        BivariateSpline : A similar class for two-dimensional spline interpolation

        Notes
        -----
        The number of data points must be larger than the spline degree `k`.

        The general form of the spline can be written as
          f[i](x) = a[i] + b[i](x - x[i]) + c[i](x - x[i])^2 + d[i](x - x[i])^3,
          i = 0, ..., n-1,
        where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.

        The unknown coefficients (a, b, c, d) define a symmetric, diagonal
        linear system of equations, Az = s, where z = b for `k` = 1 and `k` = 2,
        and z = c for `k` = 3. In each case, the coefficients defining each
        spline piece can be expressed in terms of only z[i], z[i+1],
        y[i], and y[i+1]. The coefficients are solved for using
        `jnp.linalg.solve` when `k` = 2 and `k` = 3.

        """
        # Verify inputs
        k = int(k)
        assert k in (1, 2, 3), "Order k must be in {1, 2, 3}."
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        assert len(x) == len(y), "Input arrays must be the same length."
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
        n_data = len(x)

        # Difference vectors
        h = jnp.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
        p = jnp.diff(y)  # y[i+1] - y[i]

        if coefficients is None:
            # Build the linear system of equations depending on k
            # (No matrix necessary for k=1)
            if k == 1:
                assert n_data > 1, "Not enough input points for linear spline."
                coefficients = p / h

            if k == 2:
                assert n_data > 2, "Not enough input points for quadratic spline."
                assert endpoints == "not-a-knot"  # I have only validated this
                # And actually I think it's probably the best choice of border condition

                # The knots are actually in between data points
                knots = (x[1:] + x[:-1]) / 2.0
                # We add 2 artificial knots before and after
                knots = jnp.concatenate(
                    [
                        jnp.array([x[0] - (x[1] - x[0]) / 2.0]),
                        knots,
                        jnp.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                    ]
                )
                n = len(knots)
                # Compute interval lenghts for these new knots
                h = jnp.diff(knots)
                # postition of data point inside the interval
                dt = x - knots[:-1]

                # Now we build the system natrix
                A = jnp.diag(
                    jnp.concatenate(
                        [
                            jnp.ones(1),
                            (
                                2 * dt[1:]
                                - dt[1:] ** 2 / h[1:]
                                - dt[:-1] ** 2 / h[:-1]
                                + h[:-1]
                            ),
                            jnp.ones(1),
                        ]
                    )
                )

                A += jnp.diag(
                    jnp.concatenate(
                        [-jnp.array([1 + h[0] / h[1]]), dt[1:] ** 2 / h[1:]]
                    ),
                    k=1,
                )
                A += jnp.diag(
                    jnp.concatenate([jnp.atleast_1d(h[0] / h[1]), jnp.zeros(n - 3)]),
                    k=2,
                )

                A += jnp.diag(
                    jnp.concatenate(
                        [
                            h[:-1] - 2 * dt[:-1] + dt[:-1] ** 2 / h[:-1],
                            -jnp.array([1 + h[-1] / h[-2]]),
                        ]
                    ),
                    k=-1,
                )
                A += jnp.diag(
                    jnp.concatenate([jnp.zeros(n - 3), jnp.atleast_1d(h[-1] / h[-2])]),
                    k=-2,
                )

                # And now we build the RHS vector
                s = jnp.concatenate([jnp.zeros(1), 2 * p, jnp.zeros(1)])

                # Compute spline coefficients by solving the system
                coefficients = jnp.linalg.solve(A, s)

            if k == 3:
                assert n_data > 3, "Not enough input points for cubic spline."
                if endpoints not in ("natural", "not-a-knot"):
                    print("Warning : endpoints not recognized. Using natural.")
                    endpoints = "natural"

                # Special values for the first and last equations
                zero = array([0.0])
                one = array([1.0])
                A00 = one if endpoints == "natural" else array([h[1]])
                A01 = zero if endpoints == "natural" else array([-(h[0] + h[1])])
                A02 = zero if endpoints == "natural" else array([h[0]])
                ANN = one if endpoints == "natural" else array([h[-2]])
                AN1 = (
                    -one if endpoints == "natural" else array([-(h[-2] + h[-1])])
                )  # A[N, N-1]
                AN2 = zero if endpoints == "natural" else array([h[-1]])  # A[N, N-2]

                # Construct the tri-diagonal matrix A
                A = jnp.diag(concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
                upper_diag1 = jnp.diag(concatenate((A01, h[1:])), k=1)
                upper_diag2 = jnp.diag(concatenate((A02, zeros(n_data - 3))), k=2)
                lower_diag1 = jnp.diag(concatenate((h[:-1], AN1)), k=-1)
                lower_diag2 = jnp.diag(concatenate((zeros(n_data - 3), AN2)), k=-2)
                A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

                # Construct RHS vector s
                center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
                s = concatenate((zero, center, zero))
                # Compute spline coefficients by solving the system
                coefficients = jnp.linalg.solve(A, s)

        # Saving spline parameters for evaluation later
        self.k = k
        self._x = x
        self._y = y
        self._coefficients = coefficients

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self._x, self._y, self._coefficients)
        aux_data = {"endpoints": self._endpoints, "k": self.k}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y, coefficients = children
        return cls(x, y, coefficients=coefficients, **aux_data)

    def __call__(self, x):
        """Evaluation of the spline.

        Notes
        -----
        Values are extrapolated if x is outside of the original domain
        of knots. If x is less than the left-most knot, the spline piece
        f[0] is used for the evaluation; similarly for x beyond the
        right-most point.

        """
        if self.k == 1:
            t, a, b = self._compute_coeffs(x)
            result = a + b * t

        if self.k == 2:
            t, a, b, c = self._compute_coeffs(x)
            result = a + b * t + c * t**2

        if self.k == 3:
            t, a, b, c, d = self._compute_coeffs(x)
            result = a + b * t + c * t**2 + d * t**3

        return result

    def _compute_coeffs(self, xs):
        """Compute the spline coefficients for a given x."""
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = jnp.concatenate(
                [
                    jnp.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    jnp.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = jnp.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = jnp.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]
        h = jnp.diff(knots)[ind]

        if self.k == 1:
            a = y[ind]
            result = (t, a, coefficients[ind])

        if self.k == 2:
            dt = (x - knots[:-1])[ind]
            b = coefficients[ind]
            b1 = coefficients[ind + 1]
            a = y[ind] - b * dt - (b1 - b) * dt**2 / (2 * h)
            c = (b1 - b) / (2 * h)
            result = (t, a, b, c)

        if self.k == 3:
            c = coefficients[ind]
            c1 = coefficients[ind + 1]
            a = y[ind]
            a1 = y[ind + 1]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            result = (t, a, b, c, d)

        return result

    def derivative(self, x, n=1):
        """Analytic nth derivative of the spline.

        The spline has derivatives up to its order k.

        """
        assert n in range(self.k + 1), "Invalid n."

        if n == 0:
            result = self.__call__(x)
        else:
            # Linear
            if self.k == 1:
                t, a, b = self._compute_coeffs(x)
                result = b

            # Quadratic
            if self.k == 2:
                t, a, b, c = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t
                if n == 2:
                    result = 2 * c

            # Cubic
            if self.k == 3:
                t, a, b, c, d = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t + 3 * d * t**2
                if n == 2:
                    result = 2 * c + 6 * d * t
                if n == 3:
                    result = 6 * d

        return result

    def antiderivative(self, xs):
        """
        Computes the antiderivative of first order of this spline
        """
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = jnp.concatenate(
                [
                    jnp.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    jnp.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = jnp.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = jnp.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]

        if self.k == 1:
            a = y[:-1]
            b = coefficients
            h = jnp.diff(knots)
            cst = jnp.concatenate([jnp.zeros(1), jnp.cumsum(a * h + b * h**2 / 2)])
            return cst[ind] + a[ind] * t + b[ind] * t**2 / 2

        if self.k == 2:
            h = jnp.diff(knots)
            dt = x - knots[:-1]
            b = coefficients[:-1]
            b1 = coefficients[1:]
            a = y - b * dt - (b1 - b) * dt**2 / (2 * h)
            c = (b1 - b) / (2 * h)
            cst = jnp.concatenate(
                [jnp.zeros(1), jnp.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3)]
            )
            return cst[ind] + a[ind] * t + b[ind] * t**2 / 2 + c[ind] * t**3 / 3

        if self.k == 3:
            h = jnp.diff(knots)
            c = coefficients[:-1]
            c1 = coefficients[1:]
            a = y[:-1]
            a1 = y[1:]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            cst = jnp.concatenate(
                [
                    jnp.zeros(1),
                    jnp.cumsum(
                        a * h + b * h**2 / 2 + c * h**3 / 3 + d * h**4 / 4
                    ),
                ]
            )
            return (
                cst[ind]
                + a[ind] * t
                + b[ind] * t**2 / 2
                + c[ind] * t**3 / 3
                + d[ind] * t**4 / 4
            )

    def integral(self, a, b):
        """
        Compute a definite integral over a piecewise polynomial.
        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        xs = jnp.array([a, b])
        return sign * jnp.diff(self.antiderivative(xs))

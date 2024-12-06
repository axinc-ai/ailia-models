"""Utilities to produce filters."""

import numpy as np

def _gaussian_kernel1d(sigma, order, radius):
    """
    Compute a 1-D Gaussian convolution kernel.

    From https://github.com/scipy/scipy/blob/v1.9.2/scipy/ndimage/_filters.py#L179-L207
    Copying it here since it is not part of scipy's public API.
    See https://github.com/martibosch/detectree/issues/12
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def _get_gaussian_kernel1d(sigma, *, order=0, truncate=4.0):
    """Based on scipy.ndimage.filters.gaussian_filter1d."""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # # Since we are calling correlate, not convolve, revert the kernel
    # weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = _gaussian_kernel1d(sigma, order, lw)
    return weights


def get_texture_kernel(sigma):
    """
    Get a texture kernel based on Yang et al. (2009).

    Parameters
    ----------
    sigma : numeric
        Scale parameter to build a texture kernel, based on a Gaussian on the
        X dimension and a second-derivative Gaussian in the Y dimension

    Returns
    -------
    texture_kernel : array-like
    """
    g0_kernel_arr = _get_gaussian_kernel1d(sigma, order=0)
    g2_kernel_arr = _get_gaussian_kernel1d(sigma, order=2)

    return np.dot(g2_kernel_arr.reshape(1, -1).T, g0_kernel_arr.reshape(1, -1))

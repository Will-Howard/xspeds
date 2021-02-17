import numpy as np
import matplotlib.pyplot as plt

def sph_to_cart(v):
    """Convert spherical polar coordinates to cartesian

    Args:
        v (3 element, array-like): r, theta, phi vector
    """
    r, theta, phi = v[0], v[1], v[2]
    return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

def cart_to_sph(v):
    x, y, z = v[0], v[1], v[2]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    theta = np.arctan2(np.sqrt(XsqPlusYsq), z)     # theta
    phi = np.arctan2(y, x)                           # phi
    return np.array([r, theta, phi])

def theta_to_lambda(theta):
    """Convert theta (pi / 2 - Bragg angle) to wavelength (lambda)

    Args:
        theta ([type]): (pi / 2 - Bragg angle)

    Returns:
        _lambda: wavelength in units of d
    """
    return 2 * np.cos(theta)

def lambda_to_theta(_lambda):
    """Convert wavelength (lambda) to theta (pi / 2 - Bragg angle)

    Args:
        _lambda ([type]): wavelength in units of d

    Returns:
        theta: (pi / 2 - Bragg angle)
    """
    return np.arccos(_lambda / 2)


def plot_histogram(im, n_bins=100, ax=None):
    pixel_min = min([min(r) for r in im])
    pixel_max = max([max(r) for r in im])
    d = (pixel_max - pixel_min) / n_bins

    bins = [0] * n_bins
    x = np.linspace(pixel_min, pixel_max, n_bins)
    for r in im:
        for c in r:
            _bin = (c - pixel_min) / d
            if _bin == 100.0:
                _bin = 99
            bins[int(_bin)] += 1

    if ax:
        ax.plot(x, bins)
    else:
        plt.plot(x,bins)
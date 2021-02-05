import numpy as np
import matplotlib.pyplot as plt

def pol2cart(v):
    """Convert spherical polar coordinates to cartesian

    Args:
        v (3 element, array-like): r, theta, phi vector
    """
    r, theta, phi = v
    return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

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
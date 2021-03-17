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

def dlambda_dtheta(theta):
    """Convert theta (pi / 2 - Bragg angle) to wavelength (lambda)

    Args:
        theta ([type]): (pi / 2 - Bragg angle)

    Returns:
        _lambda: wavelength in units of d
    """
    return -2 * np.sin(theta)

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
            if _bin == n_bins:
                _bin -= 1
            bins[int(_bin)] += 1

    if ax:
        ax.plot(x, bins)
    else:
        plt.plot(x,bins)


def plot_spectrum(s, xlim=[0,2]):
    """TODO allow the axes to be passed in

    Args:
        s ([type]): [description]
        xlim (list, optional): [description]. Defaults to [0,2].
    """
    fig, axs = plt.subplots(4, figsize=(10,20))
    plt.subplots_adjust(hspace=0.2)
    # intensity
    interp_lambdas = np.linspace(xlim[0], xlim[1], 1000)
    # axs[0].plot(_lambdas, intensities)
    axs[0].plot(interp_lambdas, [s.intensity(l) for l in interp_lambdas])
    axs[0].set_title("Intensity")
    # cdf
    axs[1].plot(interp_lambdas, [s.cdf(l) for l in interp_lambdas])
    axs[1].set_title("Cumulative density function")
    # inverse cdf
    p = np.linspace(0.01, 0.99, 1000)
    axs[2].plot(p, [s.inv_cdf(x) for x in p])
    axs[2].set_title("Inverse cumulative density function (useful for sampling)")
    # random sample
    N = 100
    axs[3].scatter(list(s.random_sample(N)), [1] * N, s=1)
    axs[3].set_title("Random sample")
    axs[3].set_xlim([xlim[0], xlim[1]])

    
def D(f, wrt_index, args, step_size=1e-5):
    # TODO use a library for this

    f_args = args.copy()

    f_args[wrt_index] = args[wrt_index] + 2 * step_size
    f2 = np.array(f(*f_args))

    f_args[wrt_index] = args[wrt_index] + step_size
    f1 = np.array(f(*f_args))

    f_args[wrt_index] = args[wrt_index] - step_size
    fm1 = np.array(f(*f_args))

    f_args[wrt_index] = args[wrt_index] - 2 * step_size
    fm2 = np.array(f(*f_args))


    f_prime = (-f2 + 8 * f1 - 8 * fm1 + fm2) / (12 * step_size)
    return f_prime


def J(f, args, step_size=1e-5):
    return np.stack([D(f, i, args, step_size) for i in range(len(args))]).T

def detJ(f, args, step_size=1e-5):
    return np.linalg.det(J(f, args, step_size))


def find_intersection_params(theta, a, b):
    # TODO rename this
    """Find intersection between any ray with zenith angle theta, and the line
    a + alpha * b. Returns the values of alpha at which intersection occurs
    (there will be 0, 1, or 2 of these)
    """
    t2 = np.tan(theta)**2
    A = (b[0]**2 + b[1]**2 - t2 * b[2]**2)
    B = 2 * (b[0]*a[0] + b[1]*a[1] - t2 * b[2]*a[2])
    C = (a[0]**2 + a[1]**2 - t2 * a[2]**2)

    discrim = B**2 - 4 * A * C
    if discrim < 0:
        return []
    if discrim == 0.0:
        return [(-B) / (2 * A)]
    return [(-B + np.sqrt(discrim)) / (2 * A), (-B - np.sqrt(discrim)) / (2 * A)]

def restrict_hits_pixel(hits, pixel_bounds, axis='x'):
    """TODO move this to calibration

    Args:
        hits ([type]): [description]
        pixel_bounds ([type]): [description]
        axis (str, optional): [description]. Defaults to 'x'.

    Returns:
        [type]: [description]
    """
    restricted_hits = []
    
    idx = 0 if axis.lower() == 'x' else 1
    for h in hits:
        for b in pixel_bounds:
            if b[0] <= h[idx] <= b[1]:
                restricted_hits.append(h)
                break
    return restricted_hits

def pixel_hits_to_angles(model, hits):
    """TODO move this to mock_data

    Args:
        model ([type]): [description]
        hits ([type]): [description]

    Returns:
        [type]: [description]
    """
    angle_hits = []
    for h in hits:
        theta, phi = model.plane_coords_to_angle(*model.pixel_to_plane_coords(h[0], h[1]))
        angle_hits.append((theta, phi))
    return angle_hits

def plot_theta_phi(model, hits, ax=None):
    ax = ax or plt
    angle_hits = pixel_hits_to_angles(model, hits)

    theta = [h[0] for h in angle_hits]
    phi = [h[1] for h in angle_hits]

    N = len(theta)
    s = 1000 / N

    ax.scatter(theta, phi, s=s)
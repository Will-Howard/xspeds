import xspeds.utils as utils
import numpy as np

# TODO use hypothesis (or something) for some of these


def test_pol_to_cart():
    r = np.random.uniform(0.0, 100.0)
    theta = np.random.uniform(0.0, np.pi)
    phi = np.random.uniform(-np.pi, np.pi)

    # assert converting to cartesian and back is correct to within 1e-6
    assert np.linalg.norm(np.array(utils.cart_to_sph(
        utils.sph_to_cart([r, theta, phi]))) - np.array([r, theta, phi])) < r * 1e-6
    # assert the same thing works with a numpy array
    assert np.linalg.norm(np.array(utils.cart_to_sph(
        utils.sph_to_cart(np.array([r, theta, phi])))) - np.array([r, theta, phi])) < r * 1e-6

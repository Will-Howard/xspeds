from xspeds.tests.utils import fractional_error
from xspeds.constants import IMAGE_HEIGHT, IMAGE_WIDTH
from xspeds.tests.test_spectrum import get_test_compound_spectrum
from xspeds.spectrum import LineSpectrum
from xspeds import utils
from xspeds.experimental_setup import ExperimentalSetup
import numpy as np
import scipy.integrate as integrate


def test_coord_conversion():
    w = 1.0
    m = ExperimentalSetup(sweep_angle=0.0, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=1.0, noise_std=0.2)

    for _ in range(100):
        theta = np.random.uniform(0.0, np.pi / 2)
        phi = np.random.uniform(-np.pi, np.pi)

        x, y = m.angle_to_plane_coords(theta, phi)
        # FIXME get an error when the ray ends up being cast in the negative direction, will this cause any problems??

        # assert converting to cartesian and back is correct to within 1e-6
        assert np.linalg.norm(np.array(m.plane_coords_to_angle(
            x, y)) - np.array([theta, phi])) < 1e-6


def test_azi_subtended():
    def find_angle_monte_carlo(self, theta, n_samples=10000):
        phi_samples = np.random.uniform(-np.pi, np.pi, n_samples)

        count = 0
        for phi in phi_samples:
            x, y = self.angle_to_plane_coords(theta, phi)
            if (0 < x < self.x_width and 0 < y < self.y_width):
                count += 1
        return (count / n_samples) * 2 * np.pi

    cone_angle = 0.8
    w = 0.2
    m = ExperimentalSetup(sweep_angle=0.8, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    calc_angle = m.find_azi_subtended(cone_angle)
    mc_angle = find_angle_monte_carlo(m, cone_angle)
    assert np.abs((calc_angle - mc_angle) / calc_angle) < 1e-1

    # ring entirely inside
    cone_angle = 0.2
    w = 1.0
    m = ExperimentalSetup(sweep_angle=0.0, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    calc_angle = m.find_azi_subtended(cone_angle)
    assert calc_angle == 2 * np.pi

    # cone entirely outside
    cone_angle = 0.8
    w = 0.2
    m = ExperimentalSetup(sweep_angle=0.0, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    calc_angle = m.find_azi_subtended(cone_angle)
    assert calc_angle == 0.0

# TODO mark this as slow and exclude it


def test_total_hit_prob():
    w = 0.2
    m = ExperimentalSetup(sweep_angle=0.8, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    # TODO use a less easy example
    spectrum = LineSpectrum([utils.theta_to_lambda(l)
                            for l in [0.8, 0.85]], [1, 2], [0.01, 0.01])
    
    direct_integration = integrate.dblquad(lambda y, x: m.pixel_pdf(
        spectrum, x, y), 0.0, m.x_pixels, lambda x: 0.0, lambda x: m.y_pixels, epsabs=1e-2)[0]

    total_prob = m.total_hit_probability(spectrum)[0]
    fast_total_prob = m.total_hit_probability_mod(spectrum)[0]

    print(f"total prob: {total_prob}")
    print(f"fast total prob: {fast_total_prob}")

    assert fractional_error(total_prob, direct_integration) < 1e-4
    assert fractional_error(fast_total_prob, total_prob) < 1e-4


def test_J():
    w = 0.2
    m = ExperimentalSetup(sweep_angle=0.8, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    x = np.random.uniform(0.0, 0.2)
    y = np.random.uniform(0.0, 0.2)
    numeric_J = utils.detJ(m.plane_coords_to_angle, args=[x, y])
    calc_J = m.plane_coords_to_angle_J(x, y)
    print("calc_J:", calc_J)
    print("numeric_J:", numeric_J)

    assert np.abs((calc_J - numeric_J) / numeric_J) < 1e-9


def test_pdf_vectorisation():
    w = 0.2
    m = ExperimentalSetup(sweep_angle=0.8, deviation_angles=[
                 0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    spectrum = get_test_compound_spectrum()

    # generate some random pixels
    pixels_x = np.random.uniform(0, IMAGE_WIDTH, 100)
    pixels_y = np.random.uniform(0, IMAGE_HEIGHT, 100)

    # test the individual functions used
    vec_plane_x, vec_plane_y = m.pixel_to_plane_coords(pixels_x, pixels_y)
    for i in range(len(pixels_x)):
        diff = np.array(m.pixel_to_plane_coords(pixels_x[i], pixels_y[i])) - np.array([vec_plane_x[i], vec_plane_y[i]])
        assert np.linalg.norm(diff) < 1e-8

    vec_theta, vec_phi = m.plane_coords_to_angle(vec_plane_x, vec_plane_y)
    for i in range(len(pixels_x)):
        diff = np.array(m.plane_coords_to_angle(vec_plane_x[i], vec_plane_y[i])) - np.array([vec_theta[i], vec_phi[i]])
        assert np.linalg.norm(diff) < 1e-8

    vec_J = m.plane_coords_to_angle_J(vec_plane_x, vec_plane_y)
    for i in range(len(pixels_x)):
        diff = np.array(m.plane_coords_to_angle_J(vec_plane_x[i], vec_plane_y[i])) - vec_J[i]
        assert abs(diff) < 1e-8
    #

    # test end to end
    vec_probs=m.pixel_pdf(spectrum, pixels_x, pixels_y)

    for i in range(len(pixels_x)):
        assert abs(m.pixel_pdf(
            spectrum, pixels_x[i], pixels_y[i]) - vec_probs[i]) < 1e-6

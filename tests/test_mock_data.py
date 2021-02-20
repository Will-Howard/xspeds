from xspeds.spectrum import LineSpectrum
from xspeds import utils
from xspeds.mock_data import MockData
import numpy as np
import scipy.integrate as integrate


def test_coord_conversion():
    w = 1.0
    m = MockData(sweep_angle=0.0, deviation_angles=[0.0,0.0,0.0], x_width=w, y_width=w, noise_mean=1.0, noise_std=0.2)
    
    for _ in range(100):
        theta = np.random.uniform(0.0, np.pi / 2)
        phi = np.random.uniform(-np.pi, np.pi)
        
        x, y = m.angle_to_plane_coords(theta, phi)
        # FIXME get an error when the ray ends up being cast in the negative direction, will this cause any problems??

        # assert converting to cartesian and back is correct to within 1e-6
        assert np.linalg.norm(np.array(m.plane_coords_to_angle(x, y)) - np.array([theta, phi])) < 1e-6

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
    m = MockData(sweep_angle=0.8, deviation_angles=[0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    calc_angle = m.find_azi_subtended(cone_angle)
    mc_angle = find_angle_monte_carlo(m, cone_angle)
    assert np.abs((calc_angle - mc_angle) / calc_angle) < 1e-1

    # ring entirely inside
    cone_angle = 0.2
    w = 1.0
    m = MockData(sweep_angle=0.0, deviation_angles=[0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    calc_angle = m.find_azi_subtended(cone_angle)
    assert calc_angle == 2 * np.pi

    # cone entirely outside
    cone_angle = 0.8
    w = 0.2
    m = MockData(sweep_angle=0.0, deviation_angles=[0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    calc_angle = m.find_azi_subtended(cone_angle)
    assert calc_angle == 0.0

def test_total_hit_prob():
    # TODO mark this as slow and exclude it
    w = 0.2
    m = MockData(sweep_angle=0.8, deviation_angles=[0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    spectrum = LineSpectrum([utils.theta_to_lambda(l) for l in [0.8, 0.85]], [1, 2], [0.01, 0.01])
    direct_integration = integrate.dblquad(lambda y, x: m.pixel_pdf(spectrum, x, y), 0.0, m.x_pixels, lambda x: 0.0, lambda x: m.y_pixels, epsabs=1e-2)

    assert np.abs(m.total_hit_probability(spectrum)[0] - direct_integration[0]) < 1e-3

def test_J():
    w = 0.2
    m = MockData(sweep_angle=0.8, deviation_angles=[0.0, 0.0, 0.0], x_width=w, y_width=w, noise_mean=0.0, noise_std=0.0)

    x = np.random.uniform(0.0, 0.2)
    y = np.random.uniform(0.0, 0.2)
    numeric_J = utils.detJ(m.plane_coords_to_angle, args=[x, y])
    calc_J = m.plane_coords_to_angle_J(x, y)
    print("calc_J:", calc_J)
    print("numeric_J:", numeric_J)

    assert np.abs((calc_J - numeric_J) / numeric_J) < 1e-9
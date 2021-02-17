from xspeds import utils
# from xspeds import spectrum
from xspeds.mock_data import MockData3
import numpy as np

def test_coord_conversion():

    # s = spectrum.LineSpectrum([utils.theta_to_lambda(l) for l in [0.8, 0.85]], [1, 2], [0.01, 0.01])

    w = 1.0
    m = MockData3(sweep_angle=0.0, deviation_angles=[0.0,-0.2,0.0], x_width=w, y_width=w, noise_mean=1.0, noise_std=0.2)
    
    for _ in range(10):
        theta = np.random.uniform(0.0, np.pi / 2)
        phi = np.random.uniform(-np.pi, np.pi)
        
        x, y = m.angle_to_plane_coords(theta, phi)

        # assert converting to cartesian and back is correct to within 1e-6
        assert np.linalg.norm(np.array(m.plane_coords_to_angle(x, y)) - np.array([theta, phi])) < 1e-6
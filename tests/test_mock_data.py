from xspeds import utils
# from xspeds import spectrum
from xspeds.mock_data import MockData3
import numpy as np

def test_coord_conversion():
    w = 1.0
    m = MockData3(sweep_angle=0.0, deviation_angles=[0.0,0.0,0.0], x_width=w, y_width=w, noise_mean=1.0, noise_std=0.2)
    
    for _ in range(100):
        theta = np.random.uniform(0.0, np.pi / 2)
        phi = np.random.uniform(-np.pi, np.pi)
        
        x, y = m.angle_to_plane_coords(theta, phi)
        # FIXME get an error when the ray ends up being cast in the negative direction, will this cause any problems??

        # assert converting to cartesian and back is correct to within 1e-6
        assert np.linalg.norm(np.array(m.plane_coords_to_angle(x, y)) - np.array([theta, phi])) < 1e-6
        # print("theta, phi:", theta, phi)
        # print("x,y:", x,y, phi)
        # print("theta, phi:", theta, phi)
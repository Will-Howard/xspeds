import numpy as np
from scipy.optimize import Bounds
# IMAGE_HEIGHT = 2048
# IMAGE_WIDTH = 2048

# smaller image for testing
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

X_HAT, Y_HAT, Z_HAT = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

epsilon = 0.01
WIDTH_BOUNDS = (epsilon, 1)
SWEEP_BOUNDS = (0.1, np.pi / 2 - epsilon)
DEV_ANGLE_BOUNDS = [(-np.pi / 4, np.pi / 4), (-np.pi / 4, np.pi / 4), (-np.pi / 4, np.pi / 4)]
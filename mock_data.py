# TODO probably rename this, or put it in its own module
from xspeds.constants import IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np

def gen_image(spectrum, center, boundary) -> np.ndarray:
    """Simple function to generate an image from a spectrum given
    as a list of intensities
    TODO replace this with something more sophisticated

    Args:
        spectrum (array): [description]
        center (float,float): [description]
        boundary ([type]): [description]

    Returns:
        [type]: [description]
    """
    image = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))

    intensity = sum(spectrum)
    _max_l = len(spectrum) - 1
    norm_spectrum = spectrum/intensity

    # number of points should depend on intensity, in some way...
    for _ in range(intensity*100):
        # TODO interpolate
        l = np.random.choice(range(len(spectrum)), p=norm_spectrum)
        # TODO do the displaecment correctly (using bragg's law)
        displacement = (l/_max_l)*boundary
        phi = np.random.uniform(0, 2*np.pi)

        image[int(center[0] + displacement*np.cos(phi))][int(center[1] + displacement*np.sin(phi))] += 1

    return image
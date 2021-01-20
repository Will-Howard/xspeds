# TODO probably rename this, or put it in its own module
from xspeds.spectrum import Spectrum
from xspeds.constants import IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np


def gen_image(spectrum: Spectrum, x_displacement, y_displacement, x_width, y_width,
              noise_mean=0, noise_std=0, x_pixels=IMAGE_WIDTH, y_pixels=IMAGE_HEIGHT) -> np.ndarray:
    """Generate a diffraction pattern from a spectrum.
    For now assume the image plane is normal to the axis of the crystal planes,
    the semiangle of the diffraction cone for a particular wavelentght is delta = 90 - theta
    where theta is the Bragg angle, so the radius of the ring in the image plane is
    R = L*tan(delta), where L is the distance to the crystal.
    x_displacement, y_displacement, x_width, y_width are all in units of L
    (and should therefore be quite small)

    NOTE: in a general experimental setup the image plane will not be aligned
    with the crystal planes, and the pattern for a particular wavelength on the
    image will be a general conic section
    TODO implement a more general setup

    Args:
        spectrum (Spectrum): The spectrum being diffracted throught the crystal
        x_displacement ([type]): x displacment of top left corner of detector
        y_displacement ([type]): x displacment of top left corner of detector # TODO make sure this is right
        x_width ([type]): width of the detector
        y_width ([type]): height of the detector

    Returns:
        np.ndarray: [description]
    """
    image = np.random.normal(noise_mean, noise_std,
                             (IMAGE_WIDTH, IMAGE_HEIGHT))
    x_pixel_conversion = x_pixels/x_width
    y_pixel_conversion = y_pixels/y_width

    # TODO restrict the range of phi to speed things up
    # corner_coords = [(x_displacement, y_displacement), (x_displacement, y_displacement + y_width),
    #                  (x_displacement + x_width, y_displacement), (x_displacement + x_width, y_displacement + y_width)]
    # corner_phis = list(map(lambda p: np.angle(p[0] + p[1]*1j, corner_coords)))

    # number of points should depend on intensity (TODO although maybe change this, bit cumbersome)
    for _lambda in spectrum.random_sample(int(spectrum.total_intensity*100)):
        # TODO do the displaecment correctly (using bragg's law)
        c_delta = _lambda/2  # cos(delta), _lambda is in units of d
        R = np.sqrt(1/(c_delta**2) - 1)

        phi = np.random.uniform(-np.pi, np.pi)
        x_image, y_image = R * \
            np.cos(phi) - x_displacement, R*np.sin(phi) - y_displacement

        # check if sample hits detector
        if 0 < x_image < x_width and 0 < y_image < y_width:
            x_pixel = int(x_image * x_pixel_conversion)
            y_pixel = int(y_image * y_pixel_conversion)
            # TODO make this a separate function and do clustering
            image[x_pixel][y_pixel] += 1

    return image

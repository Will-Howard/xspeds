from xspeds.spectrum import Spectrum
from xspeds.constants import IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np


class MockData:
    _hits = None

    def __init__(self,
                 x_displacement,
                 y_displacement,
                 x_width,
                 y_width,
                 x_pixels=IMAGE_WIDTH,
                 y_pixels=IMAGE_HEIGHT,
                 noise_mean=0,
                 noise_std=0,
                 ):
        self.x_displacement = x_displacement
        self.y_displacement = y_displacement
        self.x_width = x_width
        self.y_width = y_width
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def run_exposure(self, spectrum: Spectrum, time=100.0) -> np.ndarray:
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
            y_displacement ([type]): x displacment of top left corner of detector
            x_width ([type]): width of the detector
            y_width ([type]): height of the detector

        Returns:
            np.ndarray: [description]
        """
        self._hits = []

        # TODO restrict the range of phi to speed things up
        # corner_coords = [(x_displacement, y_displacement), (x_displacement, y_displacement + y_width),
        #                  (x_displacement + x_width, y_displacement), (x_displacement + x_width, y_displacement + y_width)]
        # corner_phis = list(map(lambda p: np.angle(p[0] + p[1]*1j, corner_coords)))

        # number of points should depend on intensity
        for _lambda in spectrum.random_sample(int(spectrum.total_intensity*time)):
            # delta is 90 - theta, where theta is the Bragg angle
            c_delta = _lambda/2  # cos(delta), _lambda is in units of d
            R = np.sqrt(1/(c_delta**2) - 1)  # = tan(delta)

            phi = np.random.uniform(-np.pi, np.pi)
            x_image = R * np.cos(phi) - self.x_displacement
            y_image = R * np.sin(phi) - self.y_displacement

            # check if sample hits detector
            if 0 < x_image < self.x_width and 0 < y_image < self.y_width:
                self._hits.append((x_image, y_image))

    def get_image(self):
        if self._hits is None:
            raise AssertionError("Must call run_exposure before get_image")

        image = np.random.normal(self.noise_mean, self.noise_std,
                                 (IMAGE_WIDTH, IMAGE_HEIGHT))
        x_pixel_conversion = self.x_pixels/self.x_width
        y_pixel_conversion = self.y_pixels/self.y_width

        for h in self._hits:
            x_pixel = int(h[0] * x_pixel_conversion)
            y_pixel = int(h[1] * y_pixel_conversion)
            # TODO do a gaussian (or some other) spread
            image[x_pixel][y_pixel] += 1

        return image

    def get_hits(self):
        if self._hits is None:
            raise AssertionError("Must call run_exposure before acessing hits")
        return self._hits
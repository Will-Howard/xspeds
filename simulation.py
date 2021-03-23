

from xspeds.experimental_setup import ExperimentalSetup
from xspeds.spectrum import Spectrum
import numpy as np
from copy import deepcopy

class Simulation:
    _hit_pixels = None
    _hit_lambdas = None
    has_run = False

    def __init__(self, setup: ExperimentalSetup, spectrum: Spectrum, time=100.0, n_photons=None):
        self._setup = setup
        self._spectrum = spectrum
        if n_photons is None:
            self.n_photons = int(spectrum.total_intensity * time)
        else:
            self.n_photons = n_photons
    
    def run(self):
        """Generates a mock CCD image by simulating photons hitting the detector

        Args:
            spectrum (Spectrum): The wavelength spectrum
            time (float, optional): exposure time, used along with the intensity of the spectrum
                                    to work out how many photons to simulate. Defaults to 100.0.
            n_photons ([type], optional): alternative to time, specifies exact number of photons
                                            TODO make this the number that HIT the detector, rather
                                            than the number that are simulated

        Returns:
            np.ndarray: the CCD image
        """
        self._hit_pixels = []
        self._hit_lambdas = []
        self.has_run = True

        setup = self._setup
        spectrum = self._spectrum

        # TODO restrict the range of phi to speed things up
        # corner_coords = [(x_displacement, y_displacement), (x_displacement, y_displacement + y_width),
        #                  (x_displacement + x_width, y_displacement), (x_displacement + x_width, y_displacement + y_width)]
        # corner_phis = list(map(lambda p: np.angle(p[0] + p[1]*1j, corner_coords)))

        # number of points should depend on intensity
        # TODO IMPORTANT add sin(theta) weight here
        for _lambda in spectrum.random_sample(self.n_photons):
            # theta is 90 - the Bragg angle
            c_theta = _lambda/2  # cos(theta), _lambda is in units of d
            # note this defines theta < pi / 2 (which is what I want)
            s_theta = np.sqrt(1 - c_theta**2)

            phi = np.random.uniform(-np.pi, np.pi)

            ray = np.array(
                [s_theta * np.cos(phi), s_theta * np.sin(phi), c_theta])

            # covert to plane coords
            x_image, y_image = self._setup.ray_to_plane_coords(ray)

            # check if sample hits detector
            if 0 < x_image < setup.x_width and 0 < y_image < setup.y_width:
                self._hit_pixels.append((x_image * setup.x_pixel_conversion, y_image * setup.y_pixel_conversion))
                self._hit_lambdas.append(_lambda)

    def check_has_run(self):
        if not self.has_run:
            raise AssertionError("Must run simulation before calling this function (call .run())")

    def get_hit_pixels(self):
        self.check_has_run()
        return deepcopy(self._hit_pixels)

    def get_hit_lambdas(self):
        self.check_has_run()
        return deepcopy(self._hit_lambdas)

    def get_image(self):
        self.check_has_run()

        setup = self._setup
        image = np.random.normal(setup.noise_mean, setup.noise_std,
                                 (setup.x_pixels, setup.y_pixels))

        for x_pixel, y_pixel in self._hit_pixels:
            if setup.charge_spread:
                """
                Charge spread: sample FIXED number (50) of points (don't vary with energy at this point)
                from gaussian with a (uniform) random standard deviation less than 1 pixel
                """
                # TODO make charge spread optional + variable
                n_samples = 50
                std = np.random.uniform(0.0, setup.charge_spread)
                points = np.random.multivariate_normal(
                    [x_pixel, y_pixel], std**2 * np.identity(2), n_samples)
                for p in points:
                    if 0 <= p[0] < setup.x_pixels and 0 <= p[1] < setup.y_pixels:
                        image[int(p[0])][int(p[1])] += 1 / n_samples
            else:
                image[int(x_pixel)][int(y_pixel)] += 1

        return image.T  # has to be transposed to show the right way for some reason





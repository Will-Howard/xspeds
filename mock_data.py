from xspeds.spectrum import Spectrum
from xspeds.constants import IMAGE_HEIGHT, IMAGE_WIDTH, X_HAT, Y_HAT, Z_HAT
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

from scipy.spatial.transform import Rotation as Rot

class MockData2:
    # new version with more general setup
    # z axis is aligned with cone axis
    _hits = None
    

    def __init__(self,
                 angle,
                 x_displacement,
                 y_displacement,
                 x_width,
                 y_width,
                 x_pixels=IMAGE_WIDTH,
                 y_pixels=IMAGE_HEIGHT,
                 noise_mean=0,
                 noise_std=0,
                 charge_spread=0.0,
                 ):
        self.x_disp = x_displacement
        self.y_disp = y_displacement
        self.x_width = x_width
        self.y_width = y_width
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.charge_spread = charge_spread

        rot = Rot.from_rotvec(angle * np.array([0, 1, 0]))
        self.plane_normal = rot.apply(Z_HAT) # unit normal to the image plane
        print(self.plane_normal)
        self.plane_x = rot.apply(X_HAT) # unit x in image plane
        self.plane_y = rot.apply(Y_HAT) # unit y in image plane 
        self.plane_origin = self.plane_normal + x_displacement * self.plane_x + y_displacement * self.plane_y # origin of image plane

        self.x_pixel_conversion = self.x_pixels/self.x_width
        self.y_pixel_conversion = self.y_pixels/self.y_width
        # c 0 s
        # 0 1 0
        # -s 0 c
        # calculate the normal of the image plane
        # find the origin of the image plane
        # can worry about making all the directions consistent later
        # r0 = x_disp * plane_x + ...
        # r = r0 + s u + t v
        # (r - r0) dot u == s

    def run_exposure(self, spectrum: Spectrum, time=100.0, n_photons=None) -> np.ndarray:
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
        self._hits = []

        # TODO restrict the range of phi to speed things up
        # corner_coords = [(x_displacement, y_displacement), (x_displacement, y_displacement + y_width),
        #                  (x_displacement + x_width, y_displacement), (x_displacement + x_width, y_displacement + y_width)]
        # corner_phis = list(map(lambda p: np.angle(p[0] + p[1]*1j, corner_coords)))

        # number of points should depend on intensity
        n_photons = n_photons or int(spectrum.total_intensity*time) 
        for _lambda in spectrum.random_sample(n_photons):
            # theta is 90 - the Bragg angle
            c_theta = _lambda/2  # cos(theta), _lambda is in units of d
            s_theta = np.sqrt(1 - c_theta**2)

            phi = np.random.uniform(-np.pi, np.pi)

            ray = np.array([s_theta * np.cos(phi), s_theta * np.sin(phi), c_theta])
            # a v satisfies r dot n = d, d is set to 1 so v dot n == 1/a
            a = 1/np.dot(self.plane_normal, ray)
            p_int = a * ray  # point of intersection of ray and image plane
            
            # covert to plane coords
            x_image = np.dot((p_int - self.plane_origin), self.plane_x)
            y_image = np.dot((p_int - self.plane_origin), self.plane_y)

            # check if sample hits detector
            if 0 < x_image < self.x_width and 0 < y_image < self.y_width:
                self._hits.append((x_image, y_image))

    def energy_curve(self, _lambda, n_points=1000):
        # theta is 90 - the Bragg angle
        c_theta = _lambda/2  # cos(theta), _lambda is in units of d
        s_theta = np.sqrt(1 - c_theta**2)

        phis = np.linspace(-np.pi, np.pi, n_points)

        x_out, y_out = [], []

        for phi in phis:
            ray = np.array([s_theta * np.cos(phi), s_theta * np.sin(phi), c_theta])
            # a v satisfies r dot n = d, d is set to 1 so v dot n == 1/a
            a = 1/np.dot(self.plane_normal, ray)
            p_int = a * ray  # point of intersection of ray and image plane
            
            # covert to plane coords
            x_image = np.dot((p_int - self.plane_origin), self.plane_x)
            y_image = np.dot((p_int - self.plane_origin), self.plane_y)

            # check if sample hits detector
            if 0 < x_image < self.x_width and 0 < y_image < self.y_width:
                x_out.append(x_image * self.x_pixel_conversion)
                y_out.append(y_image * self.y_pixel_conversion)

        # TODO remove artifacts due to truncation
        return x_out, y_out

    def get_image(self):
        if self._hits is None:
            raise AssertionError("Must call run_exposure before get_image")

        image = np.random.normal(self.noise_mean, self.noise_std,
                                 (IMAGE_WIDTH, IMAGE_HEIGHT))

        for h in self._hits:
            x_pixel = h[0] * self.x_pixel_conversion
            y_pixel = h[1] * self.y_pixel_conversion
            if self.charge_spread:
                """
                Charge spread: sample FIXED number (50) of points (don't vary with energy at this point)
                from gaussian with a (uniform) random standard deviation less than 1 pixel
                """
                # TODO make charge spread optional + variable
                n_samples = 50
                std = np.random.uniform(0.0,self.charge_spread)
                points = np.random.multivariate_normal([x_pixel, y_pixel], std**2 * np.identity(2), n_samples)
                for p in points:
                    if 0 <= p[0] < self.x_pixels and 0 <= p[1] < self.y_pixels:
                        image[int(p[0])][int(p[1])] += 1 / n_samples
            else:
                image[int(x_pixel)][int(y_pixel)] += 1

        return image.T  # has to be transposed to show the right way for some reason

    def get_hits(self):
        if self._hits is None:
            raise AssertionError("Must call run_exposure before acessing hits")
        return self._hits
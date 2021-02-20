from xspeds import utils
from xspeds.spectrum import Spectrum
from xspeds.constants import IMAGE_HEIGHT, IMAGE_WIDTH, X_HAT, Y_HAT, Z_HAT
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class MockData:
    _hits = None

    def __init__(self,
                 sweep_angle,
                 deviation_angles,
                 x_width,
                 y_width,
                 x_pixels=IMAGE_WIDTH,
                 y_pixels=IMAGE_HEIGHT,
                 noise_mean=0,
                 noise_std=0,
                 charge_spread=0.0,
                 ):
        self.geometry_setup(sweep_angle, deviation_angles, x_width, y_width)
        self.detector_setup(x_pixels, y_pixels, noise_mean,
                            noise_std, charge_spread)

    def geometry_setup(self, sweep_angle, deviation_angles, x_width, y_width):
        # TODO may want to change the angle convention to better reflect the expected deviations

        # perform deviation transformation as extrinsic rotations around
        # z (roll), y, x, in that order
        dev_rot = Rot.from_euler('zyx', angles=deviation_angles)
        sweep_rot = Rot.from_euler('y', sweep_angle)
        combined_rot = sweep_rot * dev_rot

        # vector pointing to center of image plane
        self.plane_center = sweep_rot.apply(Z_HAT)

        # axes of the image plane
        self.plane_normal = combined_rot.apply(Z_HAT)
        self.plane_x = combined_rot.apply(X_HAT)
        self.plane_y = combined_rot.apply(Y_HAT)

        # perpendicular (minimum) distance from source to image plane
        self.perp_dist = np.dot(self.plane_center, self.plane_normal)

        # origin of the detector, with 0 deviation the detector is normal
        # to the line connecting the source and its CENTER, not the origin
        self.plane_origin = self.plane_center - \
            (x_width * self.plane_x) / 2 - (y_width * self.plane_y) / 2

        # dimensions of the detector in units of L, the distance to the detector
        self.x_width = x_width
        self.y_width = y_width

    def detector_setup(self, x_pixels=IMAGE_WIDTH, y_pixels=IMAGE_HEIGHT, noise_mean=0.0, noise_std=0.0, charge_spread=0.0):
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.charge_spread = charge_spread

        self.x_pixel_conversion = self.x_pixels/self.x_width
        self.y_pixel_conversion = self.y_pixels/self.y_width

    def ray_to_plane_coords(self, ray):
        """get the x, y coordinates of where the ray intersects the image plane (in the plane coordinates)
        TODO change this to do (theta, phi) -> (x, y)

        Args:
            ray (np.array): unit vector in the direction of the ray
        """
        # a v satisfies r dot n = d, where d is the perp distance, so v dot n == d/a
        a = self.perp_dist/np.dot(self.plane_normal, ray)
        p_int = a * ray  # point of intersection of ray and image plane

        # covert to plane coords
        x_image = np.dot((p_int - self.plane_origin), self.plane_x)
        y_image = np.dot((p_int - self.plane_origin), self.plane_y)

        return x_image, y_image

    def angle_to_plane_coords(self, theta, phi):
        # s_theta, c_theta = np.sin(theta), np.cos(theta)
        # ray = np.array([s_theta * np.cos(phi), s_theta * np.sin(phi), c_theta])
        ray = utils.sph_to_cart([1, theta, phi])

        return self.ray_to_plane_coords(ray)

    def plane_coords_to_angle(self, x, y):
        p_int = self.plane_origin + self.plane_x * x + self.plane_y * y
        theta, phi = utils.cart_to_sph(p_int)[1:]
        return theta, phi

    def pixel_to_plane_coords(self, x_pixel, y_pixel):
        return x_pixel / self.x_pixel_conversion, y_pixel / self.y_pixel_conversion

    def plane_coords_to_pixel(self, x, y):
        return x * self.x_pixel_conversion, y * self.y_pixel_conversion

    def pdf(self, spectrum, x, y):
        """The pdf for this spectrum at x, y in PLANE COORDINATES (not pixels)
        Note that this is the pdf for the entire (infinite) plane, i.e. it is normalised to 1
        for x, y from minus inf to plus inf. You may want the pdf, given that the photon hits
        the detector, for which you will have to integrate over this function to find the
        normalisation factor

        Args:
            spectrum ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]
        """
        # WIP
        # p(x, y) = p(theta, phi)J(theta, phi)/(x, y)
        # p(theta, phi) = p(theta) / 2 pi
        # p(theta) = p(lambda)(d lambda/d theta)
        theta, _ = self.plane_coords_to_angle(x, y)
        _lambda = utils.theta_to_lambda(theta)

        p_lambda = spectrum.pdf(_lambda)
        p_theta = p_lambda * utils.D(utils.theta_to_lambda, 0, args=[theta])
        p_theta_phi = p_theta / (2 * np.pi)

        p_x_y = p_theta_phi * \
            utils.detJ(self.plane_coords_to_angle, args=[x, y])
        return np.abs(p_x_y)

    def pixel_pdf(self, spectrum, x_pixel, y_pixel):
        J = 1 / (self.x_pixel_conversion * self.y_pixel_conversion)
        return self.pdf(spectrum, *self.pixel_to_plane_coords(x_pixel, y_pixel)) * J

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
            # note this defines theta < pi / 2 (which is what I want)
            s_theta = np.sqrt(1 - c_theta**2)

            phi = np.random.uniform(-np.pi, np.pi)

            ray = np.array(
                [s_theta * np.cos(phi), s_theta * np.sin(phi), c_theta])

            # covert to plane coords
            x_image, y_image = self.ray_to_plane_coords(ray)

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
            ray = np.array(
                [s_theta * np.cos(phi), s_theta * np.sin(phi), c_theta])

            # covert to plane coords
            x_image, y_image = self.ray_to_plane_coords(ray)

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
                std = np.random.uniform(0.0, self.charge_spread)
                points = np.random.multivariate_normal(
                    [x_pixel, y_pixel], std**2 * np.identity(2), n_samples)
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

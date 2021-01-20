from typing import List, Tuple
import numpy as np
from scipy import integrate

class Spectrum:  # TODO is there any benefit to signalling it's abstractness in some way (with a decorator)
    """Abstract class representing a spectrum of radiation
    """
    total_intensity = 0

    def random_sample(self, n: int):
        """Randomly select n wavelengths from the spectrum

        Returns:
            generator: [description]
        """
        return (self.inv_cdf(np.random.uniform()) for _ in range(n))
        
    def intensity(self, _lambda):
        pass

    def cdf(self, _lambda):
        pass

    def inv_cdf(self, percentile):
        pass

class InterpolatedSpectrum(Spectrum):
    """Simple spectrum class that interpolates from a list of intensity points
    TODO the pdf and cdf don't quite correspond but this is good enough for now
    """

    def __init__(self, points: List[Tuple]):
        sorted_points = sorted(points, key=lambda p: p[0]) # sort by wavelength
        self._lambdas = np.array([p[0] for p in sorted_points])
        self._intensities = np.array([p[1] for p in sorted_points])
        self._min = self._lambdas[0]
        self._max = self._lambdas[-1]

        self.total_intensity = np.trapz(self._intensities, self._lambdas)  # TODO not sure about this

        # cache values of the cdf so it can be inverted quickly
        # TODO there is a better way to do this
        self.cached_lambdas = np.linspace(self._min, self._max, len(self._lambdas)*10)
        self.cached_cdf = [self.cdf(l) for l in self.cached_lambdas]


    def intensity(self, _lambda):
        return np.interp(_lambda, self._lambdas, self._intensities)

    def pdf(self, _lambda):
        return self.intensity(_lambda)/self.total_intensity

    def cdf(self, _lambda):
        if _lambda < self._min:
            return 0.0
        elif _lambda > self._max:
            return 1.0
        return integrate.quad(self.pdf, self._min, _lambda, points=self._lambdas)[0]

    def inv_cdf(self, p):
        return np.interp(p, self.cached_cdf, self.cached_lambdas)

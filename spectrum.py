import numpy as np
from scipy import integrate

class Spectrum:  # TODO is there any benefit to signalling it's abstractness in some way (with a decorator)
    """Abstract class representing a spectrum of radiation
    """

    def random_sample(self, n: int):
        """Randomly select n wavelengths from the spectrum

        Returns:
            generator: [description]
        """
        return (self.inv_intensity_cdf(np.random.uniform) for _ in range(n))
        
    def intensity(self, _lambda):
        pass

    def cdf(self, _lambda):
        pass

    def inv_cdf(self, _lambda):
        pass

class InterpolatedSpectrum(Spectrum):
    """Simple spectrum class that interpolates from an array of intensities
    """

    def __init__(self, _lambdas, intensities):
        self._lambdas = np.array(_lambdas)
        self._min = self._lambdas[0]
        self._max = self._lambdas[-1]
        self._intensities = np.array(intensities)

        self.total_intensity = np.trapz(self._intensities, self._lambdas)

        # cache values of the cdf so it can be inverted quickly
        # TODO there is a better way to do this
        self.cached_lambdas = np.linspace(self._min, self._max, len(self._lambdas)*10)
        self.cached_cdf = [self.cdf(l) for l in self.cached_lambdas]


    def intensity_df(self, _lambda):
        return np.interp(_lambda, self._lambdas, self._intensities)

    def pdf(self, _lambda):
        return self.intensity_df(_lambda)/self.total_intensity

    def cdf(self, _lambda):
        if _lambda < self._min:
            return 0.0
        elif _lambda > self._max:
            return 1.0
        return integrate.quad(self.pdf, self._lambdas[0], _lambda)[0]

    def inv_cdf(self, p):
        return np.interp(p, self.cached_cdf, self.cached_lambdas)

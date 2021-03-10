from typing import List, Tuple
import numpy as np
from scipy import integrate
import scipy.stats as stats
from copy import deepcopy

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
        
    def pdf(self, _lambda):
        pass

    def cdf(self, _lambda):
        pass

    def inv_cdf(self, percentile):
        pass

class CompoundSpectrum(Spectrum):
    sub_spectra = []

    def __init__(self, sub_spectra):
        self.total_intensity = sum([s.total_intensity for s in sub_spectra])
        self.sub_spectra = deepcopy(sub_spectra)

        self.cached_lambdas = np.linspace(0, 2, 5000)
        self.cached_cdf = [self.cdf(l) for l in self.cached_lambdas]

    def intensity(self, _lambda):
        return sum([s.intensity(_lambda) for s in self.sub_spectra])

    def pdf(self, _lambda):
        return self.intensity(_lambda) / self.total_intensity

    def cdf(self, _lambda):
        # average of the cdfs of the subspectra, weighted by intensity
        return sum([s.total_intensity * s.cdf(_lambda) for s in self.sub_spectra]) / self.total_intensity

    def inv_cdf(self, p):
        # return sum([s.total_intensity * s.inv_cdf(_lambda) for s in self.sub_spectra]) / self.total_intensity
        return np.interp(p, self.cached_cdf, self.cached_lambdas)

    

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
        self.cached_lambdas = None
        self.cached_cdf = None


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
        # Initialise this the first time it is called to save time in the case where only the pdf is needed
        if self.cached_lambdas == None:
            self.cached_lambdas = np.linspace(self._min, self._max, len(self._lambdas)*10)
            self.cached_cdf = [self.cdf(l) for l in self.cached_lambdas]

        return np.interp(p, self.cached_cdf, self.cached_lambdas)

class LineSpectrum(Spectrum):
    """Spectrum class that represents a series of "emission lines"
    """

    def __init__(self, _lambdas, intensities, stds):
        if not (len(_lambdas) == len(intensities) == len(stds)):
            raise ValueError("arrays must be the same length")
        self._intensities = intensities
        self._lambdas = _lambdas
        self.total_intensity = sum(intensities)
        self.line_funcs = [stats.norm(loc=_lambdas[i], scale=stds[i]) for i in range(len(_lambdas))]

        # TODO speed this up or use a different method (rejection sampling)
        self.cached_lambdas = None
        self.cached_cdf = None

    def intensity(self, _lambda):
        intensity = 0
        for i in range(len(self.line_funcs)):
            intensity += self._intensities[i] * self.line_funcs[i].pdf(_lambda)
        return intensity

    def pdf(self, _lambda):
        return self.intensity(_lambda) / self.total_intensity

    def cdf(self, _lambda):
        intensity = 0
        for i in range(len(self.line_funcs)):
            intensity += self._intensities[i] * self.line_funcs[i].cdf(_lambda)
        return intensity / self.total_intensity 

    def inv_cdf(self, p):
        if self.cached_lambdas is None:
            self.cached_lambdas = np.linspace(0, 2 * max(self._lambdas), 5000)
            self.cached_cdf = [self.cdf(l) for l in self.cached_lambdas]
        return np.interp(p, self.cached_cdf, self.cached_lambdas)
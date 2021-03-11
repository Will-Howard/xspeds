from xspeds.spectrum import LineSpectrum, CompoundSpectrum, InterpolatedSpectrum
import numpy as np
from scipy import integrate

def get_test_interp_spectrum():
    _lambdas = list(sorted(np.random.uniform(0, 2, 10)))
    points = [(l, np.random.uniform(1, 10)) for l in _lambdas]
    return InterpolatedSpectrum(points)

def get_test_line_spectrum():
    _lambdas = list(sorted(np.random.uniform(0, 2, 10)))
    intensities = [np.random.uniform(1, 10) for l in _lambdas]
    stds = [np.random.uniform(1e-4, 5e-1) for l in _lambdas]
    return LineSpectrum(_lambdas, intensities, stds)

def get_test_compound_spectrum():
    line_spec = get_test_line_spectrum()
    interp_spec = get_test_interp_spectrum()

    return CompoundSpectrum([line_spec, interp_spec])

def vectorisation_test(spectrum):
    test_lambdas = np.random.uniform(0,2, 10)

    # pdf
    vec_res = spectrum.pdf(test_lambdas)
    scalar_res = []
    # don't do list comp to make errors more readable
    for l in test_lambdas:
        scalar_res.append(spectrum.pdf(l))

    for i in range(len(scalar_res)):
        assert scalar_res[i] == vec_res[i]

    # cdf
    vec_res = spectrum.cdf(test_lambdas)
    scalar_res = []
    # don't do list comp to make errors more readable
    for l in test_lambdas:
        scalar_res.append(spectrum.cdf(l))

    for i in range(len(scalar_res)):
        assert scalar_res[i] == vec_res[i]

    # inv_cdf
    test_ps = np.random.uniform(0, 1, 100)
    vec_res = spectrum.inv_cdf(test_ps)
    scalar_res = []
    # don't do list comp to make errors more readable
    for l in test_ps:
        scalar_res.append(spectrum.inv_cdf(l))

    for i in range(len(scalar_res)):
        assert scalar_res[i] == vec_res[i]

def cdf_consistency_test(spectrum):
    test_lambdas = np.random.uniform(0, 2, 10)

    for l in test_lambdas:
        cdf = spectrum.cdf(l)
        fast_integ_pdf = integrate.quad(spectrum.pdf, 0, l)[0]
        integ_pdf = fast_integ_pdf if abs(cdf - fast_integ_pdf) < 1e-4 else integrate.quad(spectrum.pdf, 0, l, limit=500)[0]

        assert abs(cdf - integ_pdf) < 1e-4

def test_interpolated_spectrum():
    interp_spec = get_test_interp_spectrum()

    vectorisation_test(interp_spec)
    cdf_consistency_test(interp_spec)

def test_line_spectrum():
    line_spec = get_test_line_spectrum()

    vectorisation_test(line_spec)
    cdf_consistency_test(line_spec) 

def test_compound_spectrum():
    comp_spec = get_test_compound_spectrum()

    vectorisation_test(comp_spec)
    cdf_consistency_test(comp_spec) 

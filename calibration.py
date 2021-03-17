from xspeds.constants import DEV_ANGLE_BOUNDS, SWEEP_BOUNDS, WIDTH_BOUNDS
from xspeds import utils
from scipy.optimize import minimize, least_squares
from xspeds.mock_data import MockData
from xspeds.spectrum import CompoundSpectrum, InterpolatedSpectrum, LineSpectrum
import numpy as np

fit_background_intensity = 1.0
fit_background = InterpolatedSpectrum(
    [(0, fit_background_intensity), (2, fit_background_intensity)])
count = 0


def validate_params(width, sweep_angle, dev_angles, line_energies, line_intensities=None, line_widths=None):
    if width < 0:
        return False
    if not 0 < sweep_angle < np.pi / 2:
        return False
    # roll
    if not -np.pi < dev_angles[0] < np.pi:
        return False
    if any([not (- np.pi / 2 < a < np.pi / 2) for a in dev_angles[1:]]):
        return False
    if any([l < 0 for l in line_energies]):
        return False
    if line_intensities is not None and any([l < 0 for l in line_intensities]):
        return False
    if line_widths is not None and any([l < 0 for l in line_widths]):
        return False
    return True
    


def logL(spectra, setup, pixel_hits, pixel_bounds):
    # each entry of spectra, pixel_hits, pixel_bounds is to be fitted independently

    log_likelihood = 0.0
    for i in range(len(spectra)):
        spectrum, hits, bounds = spectra[i], pixel_hits[i], pixel_bounds[i]

        plane_coord_bounds = [x / setup.x_pixel_conversion for x in bounds]
        norm = setup.total_hit_probability_mod(
            spectrum, x_bounds=plane_coord_bounds)[0]

        # print(f"norm: {norm}")

        x = np.array([h[0] for h in hits])
        y = np.array([h[1] for h in hits])

        Ps = setup.pixel_pdf(spectrum, x, y)

        # if any are zero something has gone wrong
        if not all([norm] + Ps):
            return -1e10

        sum_log_p = sum(np.log(Ps))
        # print("Sum log ps: ", sum_log_p)

        log_likelihood += sum_log_p - len(Ps) * np.log(norm)
        # print(f"Log likelihood: {sum_log_p - len(Ps) * np.log(norm)}")

    # print(f"Log likelihood: {log_likelihood}")
    return log_likelihood


def logL_loss_func(width, sweep_angle, dev_angles, line_energies, line_intensities, line_widths, pixel_hits, pixel_bounds):
    # fix the background to have intensity 1, and vary the line intensities and stds

    n_lines = len(line_energies)
    # list of spectra corresponding to each group of pixels
    spectra = []
    for i in range(n_lines):
        s_line = LineSpectrum([line_energies[i]], [
                              line_intensities[i]], [line_widths[i]])
        spectra.append(CompoundSpectrum([s_line, fit_background]))

    setup = MockData(sweep_angle, dev_angles, x_width=width, y_width=width)

    log_l = logL(spectra, setup, pixel_hits, pixel_bounds)

    penalty = 0.0
    # enforce the contraint that the regions specified must entirely contain the lines
    # by adding a penalty proportional to the rms distance between line and hits
    # TODO move this to its own function
    for i in range(n_lines):
        apply_penalty = False

        lower_bound, upper_bound = pixel_bounds[i][0], pixel_bounds[i][1]
        _lambda = line_energies[i]

        n_points = 10
        y_sample = np.linspace(0, setup.y_pixels, n_points)
        x_lower = np.full(n_points, lower_bound)
        x_upper = np.full(n_points, upper_bound)

        lower_dlambda = setup.pixel_to_lambda(x_lower, y_sample) - _lambda
        lower_dlambda_sign = np.sign(lower_dlambda)
        if not np.all(lower_dlambda_sign == lower_dlambda_sign[0]):
            apply_penalty = True

        upper_dlambda = setup.pixel_to_lambda(x_upper, y_sample) - _lambda
        upper_dlambda_sign = np.sign(upper_dlambda)
        if not np.all(upper_dlambda_sign == upper_dlambda_sign[0]):
            apply_penalty = True

        if lower_dlambda_sign[0] == upper_dlambda_sign[0]:
            apply_penalty = True

        if apply_penalty:
            # penalty += 0.1 * \
            #     residuals([line_energies[i]], setup,
            #                  [pixel_hits[i]]) * log_l
            pass

    # DEBUG
    global count
    if count % 100 == 0:
        print("loss_func called with params:")
        print(f"width: {width}")
        print(f"sweep_angle: {sweep_angle}")
        print(f"dev_angles: {dev_angles}")
        print(f"intensities: {line_intensities}")
        print(f"line_widths: {line_widths}")
        print(f"Log likelihood: {log_l}")
        print(f"Loss function: {-log_l + penalty}")
    count += 1

    return -log_l + penalty


def find_maxL_params(width0, sweep_angle0, dev_angles0, line_intensities0, line_widths0, line_energies, restricted_hits, pixel_bounds, fit_small_angles=True, fit_intensities=True, fit_widths=True, return_all=False):

    def loss_func_wrapper(params):
        p_list = list(reversed(params))

        # these params are always fitted
        width = p_list.pop()
        sweep_angle = p_list.pop()

        dev_angles = [p_list.pop()]
        if fit_small_angles:
            dev_angles += [p_list.pop(), p_list.pop()]
        else:
            dev_angles += dev_angles0[1:]

        line_intensities = None
        if fit_intensities:
            line_intensities = [p_list.pop()
                                for _ in range(len(line_energies))]
        else:
            line_intensities = line_intensities0

        line_widths = None
        if fit_widths:
            line_widths = [p_list.pop() for _ in range(len(line_energies))]
        else:
            line_widths = line_widths0

        # enforce hard bounds
        if not validate_params(width, sweep_angle, dev_angles, line_energies, line_intensities, line_widths):
            return 1e10

        return logL_loss_func(width, sweep_angle, dev_angles, line_energies, line_intensities, line_widths, restricted_hits, pixel_bounds)

    p0 = [width0, sweep_angle0, dev_angles0[0]]
    if fit_small_angles:
        p0 += dev_angles0[1:]
    if fit_intensities:
        p0 += line_intensities0
    if fit_widths:
        p0 += line_widths0
    
    res = minimize(loss_func_wrapper, np.array(p0), method="Nelder-Mead", jac="3-point")
    p = res.x

    print(f"Full result: {res}")
    print(f"Resulting params: {p}")
    if return_all:
        return p

    if fit_small_angles:
        return p[:5]
    else:
        return np.append(p[:3], dev_angles0[1:])


def residuals(width, sweep_angle, dev_angles, line_energies, pixel_hits, axis='x'):
    setup = MockData(sweep_angle, dev_angles, x_width=width, y_width=width)
    residuals = []
    for i in range(len(line_energies)):
        _lambda, hits = line_energies[i], pixel_hits[i]

        # for each hit
        # calculate its position in 3-space == a
        # use the x or y axis as b
        # do find_intersection_params(theta, a, b)
        # take the smallest one, alpha = abs(min(find_intersection_params(theta, )))
        # multiply by x_pixel_conversion to get number of pixels

        theta = utils.lambda_to_theta(_lambda)
        b = setup.plane_x if axis.lower() == 'x' else setup.plane_y
        pixel_conversion = setup.x_pixel_conversion if axis.lower(
        ) == 'x' else setup.y_pixel_conversion
        for x_pixel, y_pixel in hits:
            x, y = setup.pixel_to_plane_coords(x_pixel, y_pixel)
            a = setup.plane_origin + x * setup.plane_x + y * setup.plane_y

            alphas = utils.find_intersection_params(theta, a, b)
            if len(alphas) != 0:
                residuals.append((min(alphas) * pixel_conversion))
            else:
                # if the line doesn't even intersect then we are way off
                residuals.append(setup.x_pixels)

    return np.array(residuals)

def no_azi_subtended_constraint(theta, width, sweep_angle, dev_angles):
    # TODO not even sure if I need this anymore
    setup = MockData(sweep_angle, dev_angles, x_width=width, y_width=width)

    return -setup.find_azi_subtended(theta)


def find_min_rms_params(width0, sweep_angle0, dev_angles0, line_energies, restricted_hits, fit_small_angles=True, full_result=False):
    def residuals_wrapper(params):
        p_list = list(reversed(params))

        # these params are always fitted
        width = p_list.pop()
        sweep_angle = p_list.pop()

        dev_angles = [p_list.pop()]
        if fit_small_angles:
            dev_angles += [p_list.pop(), p_list.pop()]
        else:
            dev_angles += dev_angles0[1:]

        return residuals(width, sweep_angle, dev_angles, line_energies, restricted_hits)

    p0 = [width0, sweep_angle0, dev_angles0[0]]
    if fit_small_angles:
        p0 += dev_angles0[1:]

    bounds = [WIDTH_BOUNDS, SWEEP_BOUNDS] + DEV_ANGLE_BOUNDS
    bounds = bounds[:len(p0)]
    print(f"Using bounds: {bounds}")
    flat_bounds = ([b[0] for b in bounds], [b[1] for b in bounds])

    res = least_squares(residuals_wrapper, x0=p0, bounds=flat_bounds, jac='3-point')
    p = res.x

    # print(f"Full result: {res}")
    # print(f"Cost: {res.cost}")
    if full_result:
        return res

    if fit_small_angles:
        return p[:5]
    else:
        return np.append(p[:3], dev_angles0[1:])

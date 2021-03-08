import numpy as np

# def find_photons_old(image, noise_mean=0.0, noise_std=0.0):
#     im = image.T
#     width, height = len(im), len(im[0])
#     photons = []

#     for i in range(width):
#         for j in range(height):
#             if im[i][j] > noise_mean + 5 * noise_std:
#                 # use the center of the pixel for anti aliasing
#                 photons.append((i + 0.5, j + 0.5))

#     return photons

def rescale(image, mean=None, std=None):
    mean = mean or np.mean(image)
    std = std or np.std(image)
    return (image.copy() - mean) / std

def find_photons(image, charge_mass_threshold_std, secondary_threshold_std=3, mean=None, std=None):
    # TODO use a theoretically motivated ADU value, rather than stds
    # as the primary threshold
    # TODO account for multi photon hits

    # estimate std of background from lower 95%
    image_lower_95 = image[image < np.percentile(image, 95)]

    mean = mean or np.mean(image_lower_95)
    std = std or np.std(image_lower_95)

    charge_mass_threshold = charge_mass_threshold_std * std
    # print(f"Charge mass threshold in ADU: {charge_mass_threshold}")  # DEBUG

    scaled = rescale(image, mean=mean, std=std)

    sec_hits = scaled > secondary_threshold_std #  TODO rename this

    w, h = len(image), len(image[0])

    exclude = np.zeros((w,h))

    """
    Each hits entry is:
     - the coord of the "center of mass" of the hit
     - total excess ADU (how far above the mean)
     - list of pixels included
    """
    hits = []

    def imm_adj(i, j):
        base = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        return list(filter(lambda t: 0 <= t[0] < w and 0 <= t[1] < h, base))

    def diag_adj(i, j):
        base = [(i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1), (i + 1, j + 1)]
        return list(filter(lambda t: 0 <= t[0] < w and 0 <= t[1] < h, base))

    for i in range(w):
        for j in range(h):
            if sec_hits[i][j] and not exclude[i][j]:
                """ starting at this pixel, check all immediately (not diagonal)
                pixels, if there is one with a higher ADU recenter on that.
                Then check for line, edge, box patterns"""
                i_center, j_center = i, j
                adj_adus = [((_i, _j), image[_i][_j]) for _i, _j in imm_adj(i, j) if not exclude[_i][_j]]
                max_adj = max(adj_adus, key=lambda t: t[1])

                if max_adj[1] > image[i][j]:
                    i_center, j_center = max_adj[0]
                
                inc_pixels = [(i_center, j_center)]
                # start looking for pixels to include
                # include any immediately adjacent (NOTE: this may not select for the actual most likely shapes)
                inc_adj = [(_i, _j) for _i, _j in imm_adj(i_center, j_center) if sec_hits[_i] [_j] and not exclude[_i][_j]]
                inc_pixels += inc_adj

                # look for box hits
                # TODO test that this does what I think
                diag_candidates = [(_i, _j) for _i, _j in diag_adj(i, j) if sec_hits[_i] [_j] and not exclude[_i][_j]]
                for cand in diag_candidates:
                    adj_detections = [((_i, _j) in inc_pixels) for _i, _j in imm_adj(cand[0], cand[1])]
                    if sum(adj_detections) > 1:
                        inc_pixels.append(cand)

                # exclude all included pixels from future hits
                for _i, _j in inc_pixels:
                    exclude[_i][_j] = True

                # calculate total charge mass
                charge_mass = sum([image[_i][_j] - mean for _i, _j in inc_pixels])

                # drop this hit if it is below threshold
                if charge_mass > charge_mass_threshold:
                    charge_com = sum([np.array([_i + 0.5, _j + 0.5]) * (image[_i][_j] - mean) for _i, _j in inc_pixels]) / charge_mass

                    hits.append((charge_com, charge_mass, inc_pixels))
    
    return hits
def find_photons(image, noise_mean=0.0, noise_std=0.0):
    im = image.T
    width, height = len(im), len(im[0])
    photons = []

    for i in range(width):
        for j in range(height):
            if im[i][j] > noise_mean + 5 * noise_std:
                # use the center of the pixel for anti aliasing
                photons.append((i + 0.5, j + 0.5))

    return photons
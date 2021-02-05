def find_photons(image, noise_mean=None, noise_std=None):
    im = image.T
    width, height = len(im), len(im[0])
    photons = []

    for i in range(width):
        for j in range(height):
            if im[i][j] > noise_mean + 5 * noise_std:
                photons.append((i, j))

    return photons
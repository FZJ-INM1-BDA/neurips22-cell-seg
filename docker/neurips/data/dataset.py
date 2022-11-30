import celldetection as cd
import numpy as np
from skimage import img_as_ubyte, exposure
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['normalize_img', 'normalize_channel', 'multi_norm']


def normalize_img(img, gamma_spread=17, lower_gamma_bound=.6, percentile=99.88):
    log = []
    if img.dtype.kind == 'f':  # floats
        if img.max() < 256:
            img = img_as_ubyte(img / 255)
            log.append('img_as_ubyte')
        else:
            v = 99.95
            img = cd.data.normalize_percentile(img, v)
            log.append(f'cd.data.normalize_percentile(img, {v})')
    elif img.itemsize > 1:
        img = cd.data.normalize_percentile(img, percentile)
        log.append(f'cd.data.normalize_percentile(img, {percentile})')
    mean_thresh = np.pi * gamma_spread
    if img.mean() < mean_thresh:
        gamma = (1 - ((np.cos(1 / gamma_spread * img.mean()) + 1) / 2)) * (1 - lower_gamma_bound) + lower_gamma_bound
        log.append(f'(img / 255) ** {gamma}')
        img = (img / 255) ** gamma
        img = img_as_ubyte(img)
    return img, log


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def multi_norm(img, method):
    if method == 'prov':
        img = normalize_channel(img)
    elif method == 'rand-mix' or method == 'cstm-mix':
        img0 = normalize_channel(img)
        img1, log = normalize_img(img)
        if method == 'rand-mix':
            alpha = np.random.uniform(0., 1.)
        else:
            is_grayscale = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
            alpha = 0.
            if not is_grayscale:
                if img[..., 2].mean() > 200 and img[..., 2].std() < 20:
                    alpha = 1.
            else:
                if img1.mean() < 45 and img1.std() < 33:
                    alpha = .5
        img = np.clip(alpha * img0 + (1 - alpha) * img1, 0, 255).astype(img0.dtype)
    else:
        img, log = normalize_img(img)
    return img

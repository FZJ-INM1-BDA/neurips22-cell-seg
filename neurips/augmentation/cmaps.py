from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import numpy as np


__all__ = ['get_grayscale_cmaps', 'random_grayscale_cmap']

cmaps = []
for s in dir(plt.cm):
    try:
        plt.cm.get_cmap(s)
    except ValueError:
        continue
    cmaps.append(s)

excluded_cmaps = [
    'Accent',
    'CMRmap',
    'Dark2',
    'Paired',
    'Pastel1',
    'Pastel2',
    'Set1',
    'Set2',
    'Set3',
    'flag',
    'gist_ncar',
    'gist_rainbow',
    'gist_stern',
    'gnuplot',
    'gnuplot2',
    'hsv',
    'jet',
    'nipy_spectral',
    'prism',
    'spring',
    'tab10',
    'tab20',
    'tab20b',
    'tab20c',
    'turbo',
]
excluded_cmaps += [s + '_r' for s in excluded_cmaps]
cmaps = list(set(cmaps) - set(excluded_cmaps))


def get_grayscale_cmaps():
    global cmaps
    return cmaps


def random_grayscale_cmap(image, color_maps=None):
    global cmaps
    if color_maps is None:
        color_maps = cmaps
    assert image.ndim == 2
    img = plt.cm.get_cmap(color_maps[np.random.randint(0, len(color_maps))])(image)
    return img_as_ubyte(img)[..., :3]

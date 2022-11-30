import numpy as np
import celldetection as cd
from skimage import measure
import cv2

__all__ = ['dynamic_resize']


def get_object_size_bounds(lbl):
    hmin = wmin = hmax = wmax = None
    for p in measure.regionprops(lbl):
        try:
            bb = p.bbox
            if len(bb) == 4:
                y0, x0, y1, x1 = bb
            elif len(bb) == 6:
                y0, x0, _, y1, x1, _ = bb
            else:
                raise NotImplementedError
        except ValueError as e:
            print(p.bbox)
            raise e
        h_ = y1 - y0
        w_ = x1 - x0
        if hmin is None or h_ < hmin:
            hmin = h_
        if wmin is None or w_ < wmin:
            wmin = w_
        if hmax is None or h_ > hmax:
            hmax = h_
        if wmax is None or w_ > wmax:
            wmax = w_
    return hmin, wmin, hmax, wmax


def rescale_image(img, scale, **kwargs):
    target_size = tuple(np.round(np.array(img.shape[:2]) * scale).astype('int'))
    return cv2.resize(img, target_size[::-1], **kwargs)


def dynamic_resize(img, labels, min_size=5, max_size=128, aspect_std=.05, return_scales=False, size_limit=1024 * 2):
    hmin, wmin, hmax, wmax = get_object_size_bounds(labels)
    if None in (hmin, wmin, hmax, wmax):
        if return_scales:
            return img, labels, (1., 1.)
        else:
            return img, labels

    h_limit = max(img.shape[0], size_limit)
    w_limit = max(img.shape[1], size_limit)
    h_scale_limit = h_limit / img.shape[0]
    w_scale_limit = w_limit / img.shape[0]

    h_scale_lb = min(min_size, hmin) / hmin
    w_scale_lb = min(min_size, wmin) / wmin
    h_scale_ub = min(max(max_size, hmax) / hmax, h_scale_limit)
    w_scale_ub = min(max(max_size, wmax) / wmax, w_scale_limit)

    s0 = np.random.uniform(h_scale_lb, h_scale_ub)
    s1 = np.clip(s0 + np.random.normal(0., aspect_std), w_scale_lb, w_scale_ub)
    img = rescale_image(img, (s0, s1))
    labels = rescale_image(labels, (s0, s1), interpolation=0)
    if return_scales:
        return img, labels, (s0, s1)
    return img, labels

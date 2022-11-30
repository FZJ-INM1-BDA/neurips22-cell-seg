import albumentations as A
import numpy as np
from albumentations import functional as fun
from .cmaps import random_grayscale_cmap
import cv2

__all__ = ['CMapping', 'ToRGB', 'MinimalSize']


class CMapping(A.ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, color_maps=None):
        super().__init__(always_apply=always_apply, p=p)
        self.color_maps = color_maps

    def apply(self, img, **params):
        return random_grayscale_cmap(img, color_maps=self.color_maps)


class ToRGB(A.ImageOnlyTransform):
    def apply(self, img, **params):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class MinimalSize(A.DualTransform):
    """Resize the input to the given min-height or min-width.

    Args:
        height (int): desired min-height of the output.
        width (int): desired min-width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        size = (self.height, self.width)
        scale = max(np.array(size) / np.array(img.shape[:2]))
        if scale > 1.:
            target_size = (np.array(img.shape[:2]) * scale).round().astype('int')
            return fun.resize(img, height=target_size[0], width=target_size[1], interpolation=interpolation)
        return img

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")

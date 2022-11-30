import numpy as np
import albumentations as A
from albumentations import functional as fun
from .transforms import CMapping, ToRGB, MinimalSize
import celldetection as cd
from skimage import measure
import cv2

__all__ = ['get_aug']


def get_aug(name: str, rgb: bool, crop_size: tuple):
    aug = None
    if rgb:
        if name == 'mild':
            aug = A.Compose([
                A.RandomRotate90(always_apply=True),
                A.Flip(),
                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=.2),
                    A.MedianBlur(blur_limit=3, p=.3),
                    A.Blur(blur_limit=3, p=.5)
                ], p=.4),
                A.RandomGamma(gamma_limit=(80, 120), p=.66),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(loc=0, scale=(0.01 * 255, 0.05 * 255), p=.3),
                    A.GaussNoise(p=.7)
                ], p=.1),
                A.HueSaturationValue(20, 30, 20, p=.1),
                A.ChannelShuffle(p=.1)
            ], additional_targets={
                'mask1': 'mask',
                'mask2': 'mask',
            })
    else:
        if name == 'mild':
            aug = A.Compose([
                A.RandomRotate90(),
                A.Flip(),
                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=.2),
                    A.MedianBlur(blur_limit=3, p=.3),
                    A.Blur(blur_limit=3, p=.5)
                ], p=.4),
                A.RandomGamma(gamma_limit=(80, 120), p=.66),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(loc=0, scale=(0.01 * 255, 0.05 * 255), p=.3),
                    A.GaussNoise(p=.7)
                ], p=.1),
                CMapping(p=.1, color_maps=[
                    'bone'
                ]),
                ToRGB(always_apply=True),
                A.HueSaturationValue(20, 3, 10, p=.1),
                A.ChannelShuffle(p=.2)
            ], additional_targets={
                'mask1': 'mask',
                'mask2': 'mask',
            })
    if aug is None:
        raise ValueError(name)
    return aug


def get_mild_grayscale_aug(crop_size):  # scale_limit default was .4
    return


def get_mild_rgb_aug(crop_size):
    return

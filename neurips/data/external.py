from os.path import join, dirname, basename, isfile, isdir
from os import makedirs
from glob import glob
from imageio import imread
import pandas as pd
from tqdm import tqdm
import tifffile as tif
import json
import numpy as np
import celldetection as cd
from .dataset import normalize_img
from functools import partial
from skimage.measure import regionprops


def anno2mask(anno, im):
    from pycocotools import mask as maskUtils
    rle = anno2rle(anno, im)
    return maskUtils.decode(rle)


def anno2rle(anno, im):
    from pycocotools import mask as maskUtils
    h, w = im['height'], im['width']
    seg = anno['segmentation']
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg['counts'], list):
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        rle = anno['segmentation']
    return rle


def remove_large_objects_(labels, max_size=256, constant=-1):
    for p in regionprops(labels):
        size = p.image.shape[:2]
        if max(size) > max_size:
            labels[labels == p.label] = constant


def gen_xpose(root, percentile, rm_large=False):
    images = sorted(glob(join(root, '**', '*_img*.*'), recursive=True))
    labels = [join(dirname(f), basename(f).replace('_img', '_masks')) for f in images]
    for im, la in zip(images, labels):
        img = imread(im)
        img, log = normalize_img(img, percentile=percentile)
        labels = imread(la)
        if rm_large:
            remove_large_objects_(labels, 92, -1)
        yield img, labels


gen_omnipose = partial(gen_xpose, percentile=99.9, rm_large=True)
gen_cellpose = partial(gen_xpose, percentile=99.88, rm_large=False)


def gen_sartorius(root):
    images = sorted(glob(join(root, 'train', '*.*')))
    csv_f, = glob(join(root, '**', 'train.csv'), recursive=True)
    csv = pd.read_csv(csv_f)
    for f in images:
        tag = '.'.join(basename(f).split('.')[:-1])
        sel = csv[csv.id == tag]
        rle_list = sel.annotation.tolist()
        heights = sel.height.tolist()
        widths = sel.width.tolist()

        mask = []
        for rle, h, w in zip(rle_list, heights, widths):
            m = cd.data.misc.rle2mask(rle, shape=(h, w), transpose=False)
            mask.append((m > .5).astype('uint8'))
        labels = cd.data.masks2labels(mask)
        img = imread(f)
        img, log = normalize_img(img)
        yield img, labels


def gen_livecell(root):
    images, annos, anno_map = [], [], {}
    for f_ in glob(join(root, '**', '*coco*'), recursive=True):
        j = cd.from_json(f_)
        images.extend(sorted(j['images'], key=lambda d: d['id']))
        annos.extend(sorted(j['annotations'].values(), key=lambda d: d['id']))
    for anno in annos:
        anno_map[anno['image_id']] = lst = anno_map.get(anno['image_id'], [])
        lst.append(anno)
    for im in images:
        im_annos = anno_map[im['id']]
        im_f, = glob(join(root, '**', im['file_name']), recursive=True)
        img = imread(im_f)
        img, log = normalize_img(img)
        yield img, cd.data.masks2labels([anno2mask(a, im) for a in im_annos])


def gen_bbbc038(root):
    images = sorted(glob(join(root, 'stage1_train', '*', 'images', '*.*')))
    labels = [sorted(glob(join(dirname(dirname(f)), 'masks', '*.*'))) for f in images]
    for im, la in zip(images, labels):
        img = imread(im)
        img, log = normalize_img(img, lower_gamma_bound=.4)
        masks = np.stack([imread(f) for f in la])
        lbl = cd.data.masks2labels(masks)
        yield img, lbl


def gen_bbbc039(root):
    data = cd.data.BBBC039Train(root)
    for name, img, _, lbl in data:
        img, log = normalize_img(img, percentile=99.9)
        yield img, lbl


def gen_external(**kwargs):
    for name, root in kwargs.items():
        yield from globals()[f'gen_{name}'](root)

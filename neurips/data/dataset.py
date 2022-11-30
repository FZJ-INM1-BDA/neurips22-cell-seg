from glob import glob
from os.path import join, basename, dirname, isfile, isdir
from imageio import imread, imwrite
import celldetection as cd
import numpy as np
from skimage import measure, img_as_ubyte, exposure
import cv2
from collections import OrderedDict
from ..augmentation.augmentation import dynamic_resize
from PIL import ImageFile
from skimage.measure import regionprops
from h5py import File
from ..augmentation.augmentation import rescale_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['NeurIpsTuningSet', 'NeurIpsTrainLabeled', 'normalize_img', 'Composition', 'Dataset', 'normalize_channel',
           'multi_norm', 'PseudoLabels']


def remove_fragmented_(labels, constant=-1):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_NONE
    crops = []
    for channel in np.split(labels, labels.shape[2], 2):
        crops += [(p.label, p.image) + p.bbox[:2] for p in regionprops(channel)]
    for label, crop, oy, ox in crops:
        crop.dtype = np.uint8
        r = cv2.findContours(crop, mode=mode, method=method, offset=(ox, oy))
        if len(r) == 3:  # be compatible with both existing versions of findContours
            _, c, _ = r
        elif len(r) == 2:
            c, _ = r
        else:
            raise NotImplementedError('try different cv2 version')
        if len(c) != 1:
            labels[labels == label] = constant


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


class NeurIpsTrainLabeled:
    def __init__(self, directory, cache=False, relabel=True, rgb=True, norm=True, items=None, mode='train',
                 norm_method='cstm'):
        self.img_files = []
        self.label_files = []
        if mode == 'train':
            sub_ = 'Train_Labeled'
        elif mode == 'test':
            sub_ = 'TestSet'
        else:
            raise ValueError(mode)
        lbl_names = glob(join(directory, sub_, 'labels', '*.*'))
        lbl_names_mapping = {'.'.join(basename(f).split('.')[:-1]).replace('_label', ''): f for f in lbl_names}
        for f in sorted(glob(join(directory, sub_, 'images', '*.*'))):
            self.img_files.append(f)
            f_ = lbl_names_mapping['.'.join(basename(f).split('.')[:-1])]
            self.label_files.append(f_)
        self._cache = {}
        self.cache = cache
        self.relabel = relabel
        self.rgb = rgb
        self.norm = norm
        self.norm_method = norm_method
        self.items = items
        if self.relabel:
            print("⚠️ Remove ambiguous labels.")
        if self.rgb:
            print("⚠️ Convert grayscale images to RGB.")
        if self.norm:
            print("⚠️ Normalizing all images.")

    def __getitem__(self, item):
        if self.items is not None:
            item = np.random.randint(0, len(self.img_files))
        image_f, label_f = self.img_files[item], self.label_files[item]
        img = None
        log = None
        if self.cache:
            img, lbl, log = self._cache.get(image_f, (None,) * 3)
        if img is None:
            img = imread(image_f)
            lbl = imread(label_f)
            if self.relabel:
                remove_fragmented_((lbl[..., None] if lbl.ndim == 2 else lbl))
                # lbl = measure.label(lbl)
            if self.norm:
                img = multi_norm(img, self.norm_method)
            if self.rgb:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if self.cache:
                self._cache[image_f] = img, lbl, log
        return img, lbl, (image_f, label_f, log)

    def rand(self):
        return self[np.random.randint(0, len(self))]

    def __len__(self):
        if self.items is None:
            return len(self.img_files)
        return self.items

    def unique_shapes(self):
        s = []
        for img, labels, names in self:
            s.append(img.shape)
        return set(s)

    def unique_dtypes(self):
        s = []
        for img, labels, names in self:
            s.append(img.dtype)
        return set(s)


class NeurIpsTuningSet:
    def __init__(self, directory, norm=True, rgb=True, norm_method='cstm'):
        self.img_files = glob(join(directory, 'TuningSet', '*'))
        self.rgb = rgb
        self.norm = norm
        self.norm_method = norm_method

    def __getitem__(self, item):
        image_f = self.img_files[item]
        img = imread(image_f)
        if self.norm:
            img = multi_norm(img, self.norm_method)
            if self.rgb:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img, image_f

    def __len__(self):
        return len(self.img_files)


class Composition:
    def __init__(self, *datasets, handlers=None):
        self.handlers = handlers
        self.datasets = datasets
        self.lengths = np.array([len(d) for d in self.datasets])

    def __len__(self):
        return np.sum(self.lengths)

    def __getitem__(self, item):
        cs = np.cumsum(self.lengths)
        if item >= cs[-1]:
            raise IndexError
        dataset_idx = np.argmax(cs > item)
        item = item - np.sum(self.lengths[:dataset_idx])
        ds = self.datasets[dataset_idx]
        ret = ds[item]
        if self.handlers is not None:
            handler = self.handlers[dataset_idx]
            if handler is not None:
                ret = handler(*ret)
        return ret, (dataset_idx, item, type(ds))


def special_dynamic_resize(img, labels, score_maps, dyn_resize_min=5, dyn_resize_max=192,
                           dyn_resize_aspect_std=.05):
    img, labels, (s0, s1) = dynamic_resize(img, labels, dyn_resize_min, dyn_resize_max,
                                           aspect_std=dyn_resize_aspect_std, return_scales=True)
    score_maps = rescale_image(score_maps, (s0, s1))
    return img, labels, score_maps


class Dataset:
    def __init__(self, data, config, gray_transforms=None, rgb_transforms=None, items=None, size=None):
        self.gray_transforms = gray_transforms
        self.rgb_transforms = rgb_transforms
        assert len(self.gray_transforms.additional_targets), 'Add additional mask targets'
        assert len(self.rgb_transforms.additional_targets), 'Add additional mask targets'

        self.gen = cd.data.CPNTargetGenerator(
            samples=config.samples,
            order=config.order,
            max_bg_dist=config.bg_fg_dists[0],
            min_fg_dist=config.bg_fg_dists[1],
            # flag_fragmented=True
        )
        self._items = items or len(data)
        self.data = data
        self.size = size
        self.channels = config.in_channels
        self.conf = config

        self.dyn_resize_min = config.dyn_resize_min
        self.dyn_resize_max = config.dyn_resize_max
        self.dyn_resize_aspect_std = config.dyn_resize_aspect_std

    def __len__(self):
        return self._items

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError('Index out of bounds.')
        item = item % len(self.data)

        # Get image and labels
        r, (dataset_idx, dataset_item_idx, dataset_type) = self.data[item]
        if len(r) == 3:
            img, labels, names = r
            num = labels.max()
            uncertainty = np.zeros((num, 4), dtype='float32')
            scores = np.ones(num, dtype='float32')
            score_maps = np.zeros(img.shape[:2] + (1,), dtype='float32')  # only used for background, hence zero
            # name = names[0]
        elif len(r) > 6:
            img, labels, __contours, scores, __boxes, uncertainty, score_maps, _ = r
            # name = img_f
        else:
            raise ValueError

        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        img, labels, score_maps = cd.data.random_crop(img, labels, score_maps, height=2048, width=2048)
        if np.random.rand() > .8:
            img, labels, score_maps = special_dynamic_resize(img, labels, score_maps,
                                                             dyn_resize_min=self.dyn_resize_min,
                                                             dyn_resize_max=self.dyn_resize_max,
                                                             dyn_resize_aspect_std=self.dyn_resize_aspect_std)

        # Normalize intensities
        img, labels = np.copy(img).squeeze(), np.copy(labels)
        labels = labels.astype('int32')

        # Crop to avoid larger augmentations
        img, labels, score_maps = cd.data.random_crop(img, labels, score_maps, height=1024, width=1024)

        # Optionally transform
        aug_fn = self.gray_transforms if img.ndim == 2 else self.rgb_transforms
        if aug_fn is not None:
            r = aug_fn(image=img, mask1=labels, mask2=score_maps)
            img, labels, score_maps = r['image'], r['mask1'], r['mask2']

        # Optionally crop
        if self.size is not None:
            h, w = self.size
            img, labels, score_maps = cd.data.random_crop(img, labels, score_maps, height=h, width=w)

        # mode = 'reflect'  # TODO
        mode = 'constant'  # TODO
        if np.any(np.array(img.shape[:2]) < self.size):
            img = np.pad(img, [[0, self.size[i] - img.shape[i]] if i < 2 else [0, 0] for i in range(img.ndim)],
                         mode=mode)
            labels = np.pad(labels,
                            [[0, self.size[i] - labels.shape[i]] if i < 2 else [0, 0] for i in range(labels.ndim)],
                            mode=mode)
            score_maps = np.pad(score_maps,
                                [[0, self.size[i] - score_maps.shape[i]] if i < 2 else [0, 0] for i in
                                 range(score_maps.ndim)],
                                mode=mode)

        # Ensure channels exist
        if labels.ndim == 2:
            labels = labels[..., None]

        uni = np.unique(labels)
        uni = sorted(list(uni[uni > 0] - 1))
        scores = scores[uni]
        uncertainty = uncertainty[uni]

        # Relabel to ensure that N objects are marked with integers 1..N
        cd.data.segmentation.fill_label_gaps_(labels)
        remove_fragmented_(labels)

        # Feed labels to target generator
        gen = self.gen
        gen.feed(labels=labels)

        if img.ndim == 2:
            img = img[..., None]
        if score_maps.ndim == 2:
            score_maps = score_maps[..., None]

        # Map image to range 0..1
        img = img / 255

        # Return as dictionary
        return OrderedDict({
            'inputs': img.astype('float32'),
            'labels': gen.reduced_labels,
            'fourier': (gen.fourier.astype('float32'),),
            'locations': (gen.locations.astype('float32'),),
            'sampled_contours': (gen.sampled_contours.astype('float32'),),
            'sampling': (gen.sampling.astype('float32'),),
            'targets': labels,
            'teacher_scores': (scores.astype('float32'),),
            'teacher_score_maps': score_maps.astype('float32'),
            'teacher_uncertainty': (uncertainty.astype('float32'),),
        })


class PseudoLabels:
    def __init__(self, search_pattern: str, items=None, norm=True, norm_method='cstm'):
        self.files = sorted(glob(search_pattern))
        self.items = items
        self.norm = norm
        self.norm_method = norm_method

    def __len__(self):
        if self.items is None:
            return len(self.files)
        return self.items

    def __getitem__(self, item):
        if self.items is not None:  # random in subset mode
            if item >= self.items:
                raise IndexError
            item = np.random.randint(0, len(self.files))
        f = self.files[item]
        with File(f, 'r') as h:
            img = h['image'][:]
            if self.norm:
                img = multi_norm(img, self.norm_method)
            contours = h['contours'][:]
            scores = h['scores'][:]
            boxes = h['boxes'][:]
            uncertainty = h['uncertainty'][:]
            score_maps = h['score_maps'][:]
        labels = cd.data.contours2labels(contours, img.shape[:2])
        return img, labels, contours, scores, boxes, uncertainty, score_maps, (f,)

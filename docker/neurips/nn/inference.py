import warnings
import celldetection as cd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision.ops.boxes import box_iou, remove_small_boxes
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from . import functional as fun
from itertools import product

__all__ = ['CpnInference']


def get_tiling_slices(size, crop_size, strides):
    assert isinstance(size, (tuple, list))
    slices, shape = [], []
    for axis in range(len(size)):
        if crop_size[axis] >= size[axis]:
            tl = [size[axis]]
        else:
            tl = range(crop_size[axis], max(2, 1 + int(np.ceil(size[axis] / strides[axis]))) * strides[axis],
                       strides[axis])
        axis_slices = []
        for t in tl:
            stop = min(t, size[axis])
            axis_slices.append(slice(max(0, stop - crop_size[axis]), stop))
        slices.append(axis_slices), shape.append(len(tl))
    return product(*slices), shape


class TileLoader:
    def __init__(self, img, mask=None, transforms=None, reps=8, crop_size=(512, 512), strides=(512 - 128, 512 - 128)):
        """

        Notes:
            - if mask is used, batch_size may be smaller, as items may be dropped

        Args:
            img: Array[h, w, ...] or Tensor[..., h, w].
            mask: Always as Array[h, w, ...]
            transforms:
            reps:
            crop_size:
            strides:
        """
        if isinstance(img, torch.Tensor):
            size = img.shape[-len(crop_size):]
            self.slice_prefix = (...,)
        else:
            size = img.shape[:len(crop_size)]
            self.slice_prefix = ()
        self.slices, self.num_slices_per_axis = get_tiling_slices(size, crop_size, strides)
        self.slices = list(self.slices)
        self.reps = reps
        self.img = img
        self.transforms = transforms
        self.mask = mask

    def __len__(self):
        return len(self.slices) * self.reps

    def __getitem__(self, item):
        slice_idx = item // self.reps
        rep_idx = item % self.reps
        slices = self.slices[slice_idx]
        if self.mask is not None:
            mask_crop = self.mask[slices]
            if not np.any(mask_crop):
                return None
        crop = self.img[self.slice_prefix + slices]
        meta = ()
        if self.transforms is not None:
            crop, meta = self.transforms(crop, rep_idx)
        return crop, (slice_idx, rep_idx, slices) + meta


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], 0)
    meta = [b[1] for b in batch]
    return images, meta


def ensure_float32(v: list):
    return [(v_ if v_.element_size() >= 4 else v_.to(torch.float32)) for v_ in v]


def apply_rep_voting(contours, scores, boxes, uncertainties, iou_threshold, reps, min_vote_fraction, weighted=True,
                     method='vote'):
    assert method in ('vote', 'mean')

    if len(contours) <= 0:
        return contours, scores, boxes, uncertainties, scores

    nms_weight = scores * (1. - torch.sigmoid(uncertainties.mean(1))) if weighted else scores
    keep = torch.ops.torchvision.nms(boxes, nms_weight, iou_threshold=iou_threshold)
    boxes, scores, contours, uncertainties, nms_weight = [i[keep] for i in
                                                          (boxes, scores, contours, uncertainties, nms_weight)]
    return contours, scores, boxes, uncertainties, nms_weight


class CpnInference(nn.Module):
    def __init__(self, model, transforms, reps=8, window_removal_pad=8,
                 crop_size=(512, 512), strides=(384, 384), weighted_tile_nms=True, weight_final_nms=True,
                 tiled_final_nms=True,
                 amp=False, nms_crop_size=(8192, 8192), nms_strides=(8000, 8000), min_vote_fraction=.2,
                 voting_method='vote'):
        """

        Notes:
            - Default device for all operation is model.device. If it is initially on CPU, everything will run on CPU.
            - Output is on CPU. This allows for outputs that do not fit on GPU memory.
            - NMS runs on GPU (possibly tiled). It is beneficial if ``model`` is only stored inside an object
                of this class, so that it can be removed from its current device to free space for final NMS.
            - This class is made to process single images (possibly in tiles with batch_size >= 1).

        Args:
            model:
            transforms:
            reps:
            removal_pad:
            window_removal_pad:
            overlap:
            crop_size:
            strides:
            weight_final_nms:
            tiled_final_nms:
            amp:
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = cd.get_device(self.model)
        self.transforms = transforms
        self.plan = [
            {'rot90': 0, 'transpose': False},  # first/last 4 can be in one batch, even if not square img
            {'rot90': 1, 'transpose': True},
            {'rot90': 3, 'transpose': True},
            {'rot90': 2, 'transpose': False},
            {'rot90': 0, 'transpose': True},
            {'rot90': 1, 'transpose': False},
            {'rot90': 2, 'transpose': True},
            {'rot90': 3, 'transpose': False}
        ]
        self.voting_method = voting_method
        self.reps = reps
        self.window_removal_pad = window_removal_pad
        self.gpu_st = cd.GpuStats()
        self.min_vote = min_vote_fraction
        self.crop_size = crop_size
        self.strides = strides
        self.gpu_st = cd.GpuStats()
        self.weight_final_nms = weight_final_nms  # first submission was without this
        self.weighted_tile_nms = weighted_tile_nms
        self.tiled_final_nms = tiled_final_nms
        self.final_nms = True
        self.nms_crop_size = nms_crop_size
        self.nms_strides = nms_strides
        self.amp = amp
        self.verbose = True
        self.loader_conf = cd.Config(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=4
        )
        self.min_box_size = 1.
        self.norm_div = 255
        self._warned_norm = False
        self._warned_unc = False

    def process_patch(self, x):
        x = self.to_tensor(x)
        with autocast(self.amp):
            y = self.model(x.to(self.device))
        contours = ensure_float32(y['contours'])
        scores = ensure_float32(y['scores'])
        boxes = ensure_float32(y['boxes'])
        if 'box_uncertainties' in y.keys():
            uncertainty = ensure_float32(y['box_uncertainties'])
        else:
            if not self._warned_unc:
                warnings.warn('Could not find box uncertainties. Using 1. - scores now.')
                self._warned_unc = True
            uncertainty = [1. - s for s in scores]
        return contours, scores, boxes, uncertainty

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            if x.ndim == 2:
                x = x[None]
            if x.ndim == 3:
                x = x[None]
            return x
        if x.ndim == 2:
            x = x[..., None]
        x = cd.data.to_tensor(x.copy(), transpose=True, dtype=torch.float32)
        if getattr(self.model, 'transform', None) is None:
            x = x / 127.5 - 1
            if not self._warned_norm:
                warnings.warn('model.transform was not found. Applying (x/127.5-1) as normalization.')
                self._warned_norm = True
        else:
            x = x / self.norm_div
        return x

    def _forward_tiled(self, x, is_np, size, mask=None):

        def transforms(inputs, rep):
            aug_plan = self.plan[rep % len(self.plan)]

            if self.transforms is not None:
                x_ = self.transforms(image=inputs)['image']
            else:
                x_ = inputs

            rotation_size = None
            if aug_plan['rot90']:
                if is_np:
                    x_ = np.rot90(x_, aug_plan['rot90'])
                    rotation_size = x_.shape[:2]
                else:
                    x_ = cd.ops.rot90_2d(x_, aug_plan['rot90'])
                    rotation_size = x_.shape[-2:]
            if aug_plan['transpose']:
                if is_np:
                    x_ = np.transpose(x_, (1, 0, 2))
                else:
                    x_ = cd.ops.transpose2d(x_)
            x_ = self.to_tensor(x_)
            return x_, (aug_plan, rotation_size,)

        tile_loader = TileLoader(x, mask=mask, crop_size=self.crop_size, strides=self.strides, reps=self.reps,
                                 transforms=transforms)
        data_loader = DataLoader(
            tile_loader,
            batch_size=self.loader_conf.batch_size,
            num_workers=self.loader_conf.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            persistent_workers=False,
            pin_memory=self.loader_conf.pin_memory,
            **({'prefetch_factor': self.loader_conf.get('prefetch_factor', 2)} if self.loader_conf.num_workers else {})
        )

        tiled_contours = {i: [] for i in range(np.prod(tile_loader.num_slices_per_axis))}
        tiled_scores = {i: [] for i in range(np.prod(tile_loader.num_slices_per_axis))}
        tiled_boxes = {i: [] for i in range(np.prod(tile_loader.num_slices_per_axis))}
        tiled_uncertainties = {i: [] for i in range(np.prod(tile_loader.num_slices_per_axis))}

        tq = tqdm(data_loader) if self.verbose else data_loader
        for batch, meta in tq:
            h_tiles, w_tiles = tile_loader.num_slices_per_axis
            outputs = self.process_patch(batch)
            if self.verbose:
                tq.desc = str(self.gpu_st) + f', shape: {x.shape}, tiles: {tile_loader.num_slices_per_axis}'

            for n, (contours, scores, boxes, uncertainties) in enumerate(zip(*outputs)):
                slice_idx, rep_idx, slices, k, rot_size = meta[n]
                h_i, w_i = np.unravel_index(slice_idx, tile_loader.num_slices_per_axis)
                h_start, w_start = [s.start for s in slices]

                if k['transpose']:
                    contours = fun.coords_transpose(contours)
                    boxes = fun.boxes_transpose(boxes)
                if k['rot90']:
                    num_rot = 4 - k['rot90']
                    contours = fun.coords_rot90(contours, *rot_size, k=num_rot)
                    boxes = fun.boxes_rot90(boxes, *rot_size, k=num_rot)

                # Remove small boxes
                keep = remove_small_boxes(boxes, self.min_box_size)
                contours, scores, boxes, uncertainties = (c[keep] for c in (contours, scores, boxes, uncertainties))

                # Remove partial detections to avoid tiling artifacts
                keep = fun.remove_border_contours(contours, batch.shape[-2:], self.window_removal_pad,
                                                  top=h_i > 0,
                                                  right=w_i < (w_tiles - 1),
                                                  bottom=h_i < (h_tiles - 1),
                                                  left=w_i > 0)
                contours, scores, boxes, uncertainties = (c[keep] for c in (contours, scores, boxes, uncertainties))

                # Add offset
                contours[..., 1] += h_start
                contours[..., 0] += w_start
                boxes[..., [0, 2]] += w_start
                boxes[..., [1, 3]] += h_start

                tiled_contours[slice_idx].append(contours.cpu())
                tiled_scores[slice_idx].append(scores.cpu())
                tiled_boxes[slice_idx].append(boxes.cpu())
                tiled_uncertainties[slice_idx].append(uncertainties.cpu())

        all_contours, all_scores, all_boxes, all_weights, all_uncertainties = [], [], [], [], []
        for slice_idx in tiled_contours.keys():
            contours = tiled_contours[slice_idx]
            if len(contours) <= 0:
                continue
            scores, boxes = tiled_scores[slice_idx], tiled_boxes[slice_idx]
            uncertainties = tiled_uncertainties[slice_idx]
            contours, scores, boxes, uncertainties, nms_weight = apply_rep_voting(
                *[torch.cat(c).to(self.device) for c in (contours, scores, boxes, uncertainties)],
                iou_threshold=self.model.nms_thresh, reps=self.reps, min_vote_fraction=self.min_vote,
                weighted=self.weighted_tile_nms, method=self.voting_method)
            all_contours.append(contours.cpu()), all_scores.append(scores.cpu()), all_boxes.append(boxes.cpu())
            all_weights.append(nms_weight.cpu())
            all_uncertainties.append(uncertainties.cpu())

        # Gather all results
        if len(all_contours) <= 0:
            return (
                torch.empty((0, 0, 4), device=self.device),
                torch.empty((0,), device=self.device),
                torch.empty((0, 4), device=self.device),
                torch.empty((0,), device=self.device),
                torch.empty((0,), device=self.device),
            )

        contours, boxes, scores = torch.cat(all_contours), torch.cat(all_boxes), torch.cat(all_scores)
        uncertainties = torch.cat(all_uncertainties)
        nms_weights = torch.cat(all_weights)

        # Final nms (takes care of inter-tile redundancies)
        if self.final_nms and max(tile_loader.num_slices_per_axis) > 1:
            print('inter-tile nms', flush=True)
            nms_score = nms_weights if self.weight_final_nms else scores
            if self.tiled_final_nms and max(size) > 4000:
                keep = fun.tiled_nms(boxes=boxes, scores=nms_score, iou_threshold=self.model.nms_thresh, size=size,
                                     device=None, crop_size=self.nms_crop_size, progress=self.verbose,
                                     strides=self.nms_strides)
            else:
                keep = torch.ops.torchvision.nms(boxes, nms_score, iou_threshold=self.model.nms_thresh)
            contours, scores, boxes, nms_weights, uncertainties = (c[keep] for c in (
                contours, scores, boxes, nms_weights, uncertainties))
        else:
            print('skipping inter-tile nms', flush=True)
        return contours, scores, boxes, uncertainties, nms_weights

    def forward(self, x, mask=None):
        is_np = isinstance(x, np.ndarray)
        if is_np:
            size = x.shape[:2]
        else:
            size = x.shape[-2:]
        with torch.no_grad():
            contours, scores, boxes, uncertainty, weights = self._forward_tiled(x, is_np, size, mask=mask)
        return contours, scores, boxes, uncertainty, weights

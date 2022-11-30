import torch
from tqdm import tqdm
import numpy as np
from torch import Tensor
from typing import List, Tuple

__all__ = ['tiled_nms', 'batched_box_nmsi']


def batched_box_nmsi(
        boxes: List[Tensor], scores: List[Tensor], iou_threshold: float
) -> Tuple[List[Tensor], ...]:
    assert len(scores) == len(boxes)
    keeps = []
    for con, sco in zip(boxes, scores):
        indices = torch.ops.torchvision.nms(con, sco, iou_threshold=iou_threshold)
        keeps.append(indices)
    return keeps


def tiled_nms(boxes, scores, size, iou_threshold, crop_size=(8192, 8192), strides=(8000, 8000), progress=True,
              device=None):
    """Tiled NMS.

    TODO: Improve efficiency! Easy target to improve inference speed further

    Notes:
        - overlap can be small or 0, as all boxes are included that share any point with the tile area

    Args:
        boxes: Tensor[num, 4] in (x0, y0, x1, y1) format.
        scores:
        size:
        iou_threshold:
        crop_size:
        strides:
        progress:
        device:
        default_device

    Returns:

    """
    default_device = boxes.device
    keep = torch.ones(scores.shape, dtype=torch.bool, device=default_device)
    h_tiles = range(crop_size[0], max(2, 1 + int(np.ceil(size[0] / strides[0]))) * strides[0],
                    strides[0])
    w_tiles = range(crop_size[1], max(2, 1 + int(np.ceil(size[1] / strides[1]))) * strides[1],
                    strides[1])
    tq = tqdm(total=len(h_tiles) * len(w_tiles), desc='Tiled NMS') if progress else None
    for h_i, h in enumerate(h_tiles):
        h_stop = min(h, size[0])
        h_start = max(0, h_stop - crop_size[0])
        for w_i, w in enumerate(w_tiles):
            w_stop = min(w, size[1])
            w_start = max(0, w_stop - crop_size[1])

            bw = boxes[..., [0, 2]]
            bh = boxes[..., [1, 3]]
            sel_w = torch.any(bw >= w_start, dim=-1) & torch.any(bw <= w_stop, dim=-1)
            sel_h = torch.any(bh >= h_start, dim=-1) & torch.any(bh <= h_stop, dim=-1)
            sel = sel_w & sel_h & keep

            sel_keep_indices = torch.ops.torchvision.nms(boxes[sel].to(device), scores[sel].to(device),
                                                         iou_threshold=iou_threshold).to(default_device)
            sel_keep = torch.zeros(keep[sel].shape, dtype=torch.bool, device=default_device)
            sel_keep[sel_keep_indices] = True
            keep[sel] = sel_keep
            if tq is not None:
                tq.update(1)
    return keep

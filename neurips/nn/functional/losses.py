import torch
from torch import Tensor
import torchvision.ops.boxes as bx
from typing import Tuple
import numpy as np
import celldetection as cd


def _pairwise_box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = bx.box_area(boxes1)
    area2 = bx.box_area(boxes2)

    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])  # Tensor[N,2]
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])  # Tensor[N,2]

    wh = bx._upcast(rb - lt).clamp(min=0)  # Tensor[N,2]
    intersection = torch.prod(wh, dim=1)  # Tensor[N]
    union = area1 + area2 - intersection
    return intersection, union


def pairwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    inter, union = _pairwise_box_inter_union(boxes1, boxes2)
    return torch.abs(inter / union)


def pairwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    inter, union = _pairwise_box_inter_union(boxes1, boxes2)
    iou = inter / union
    lti = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    whi = bx._upcast(rbi - lti).clamp(min=0)  # Tensor[N,2]
    areai = torch.prod(whi, dim=1)
    return iou - (areai - union) / areai


def soft_l1l2(inputs, targets, switch=7, alpha_exp=2, reduction='mean'):
    diff = torch.abs(inputs - targets)
    alpha = torch.clip(diff / switch, 0., 1.)
    if alpha_exp is not None:
        alpha = alpha ** alpha_exp
    loss = diff * (1 - alpha) + alpha * torch.square(diff)
    loss = cd.ops.reduce_loss(loss, reduction=reduction)
    return loss


def soft_iou_l1l2(inputs, targets, boxes, boxes_targets, reduction='mean', generalized=True):
    diff = torch.abs(inputs - targets)  # Tensor[n, samples, 2]
    if generalized:
        iou = .5 * (1 + pairwise_generalized_box_iou(boxes, boxes_targets))  # Tensor[n]
    else:
        iou = pairwise_box_iou(boxes, boxes_targets)  # Tensor[n]
    iou = iou[:, None, None]
    loss = diff * iou + (1 - iou) * torch.square(diff)
    loss = cd.ops.reduce_loss(loss, reduction=reduction)
    return loss


def iou_loss(boxes, boxes_targets, reduction='mean', generalized=True, method='linear', min_size=None):
    if min_size is not None:  # eliminates invalid boxes
        keep = bx.remove_small_boxes(boxes, min_size)
        boxes, boxes_targets = (c[keep] for c in (boxes, boxes_targets))

    if generalized:
        iou = pairwise_generalized_box_iou(boxes, boxes_targets)  # Tensor[n]
    else:
        iou = pairwise_box_iou(boxes, boxes_targets)  # Tensor[n]

    if method == 'log':
        if generalized:
            iou = iou * .5 + .5
        loss = -torch.log(iou + 1e-8)
    elif method == 'linear':
        loss = 1 - iou
    else:
        raise ValueError

    loss = cd.ops.reduce_loss(loss, reduction=reduction)
    return loss


def box_npll_loss(uncertainty, boxes, boxes_targets, epsilon=1e-8, reduction='mean', sigmoid=False, min_size=None,
                  sigmoid_factor=10.):
    """NPLL.

    References:
        https://arxiv.org/abs/2006.15607

    Args:
        uncertainty: Tensor[n, 4].
        boxes: Tensor[n, 4].
        boxes_targets: Tensor[n, 4].
        epsilon: Epsilon.
        reduction: Loss reduction.

    Returns:
        Loss.
    """
    if min_size is not None:  # eliminates invalid boxes
        keep = bx.remove_small_boxes(boxes, min_size)
        boxes, boxes_targets, uncertainty = (c[keep] for c in (boxes, boxes_targets, uncertainty))
    delta_sq = torch.square((torch.sigmoid(uncertainty) * sigmoid_factor) if sigmoid else uncertainty)  # sigfactor=1.
    a = torch.square(boxes - boxes_targets) / (2 * delta_sq + epsilon)
    b = 0.5 * torch.log(delta_sq + epsilon)
    iou = pairwise_box_iou(boxes, boxes_targets)  # Tensor[n]
    loss = iou * ((a + b).sum(dim=1) + 2 * np.log(2 * np.pi))
    loss = cd.ops.reduce_loss(loss, reduction=reduction)
    return loss


def tversky_loss(inputs, targets, alpha=.5, beta=.5, sigmoid=True, smooth=1.):
    if sigmoid:
        inputs = inputs.sigmoid()
    a = (inputs * targets).sum()
    loss = (a + smooth) / (smooth + a + alpha * (inputs * (1 - targets)).sum() + beta * ((1 - inputs) * targets).sum())
    return loss


def focal_tversky_loss(inputs, targets, alpha=.5, beta=.5, gamma=1.5, sigmoid=True, smooth=1.):
    loss = tversky_loss(inputs, targets, alpha=alpha, beta=beta, gamma=gamma, sigmoid=sigmoid, smooth=smooth)
    return (1 - loss) ** gamma

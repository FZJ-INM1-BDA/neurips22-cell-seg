import warnings
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List
from celldetection.util.util import add_to_loss_dict, reduce_loss_dict, fetch_model
from celldetection.models.commons import ScaledTanh, ReadOut
from celldetection.ops.commons import downsample_labels
from celldetection.util import lookup_nn
from .functional.nms import batched_box_nmsi
from celldetection.ops.cpn import rel_location2abs_location, fouriers2contours, scale_contours, scale_fourier, \
    order_weighting, resolve_refinement_buckets
from .unet import U22, SlimU22, WideU22, ResUNet, ResNet50UNet, ResNet34UNet, ResNet18UNet, \
    ResNet101UNet, ResNeXt50UNet, ResNet152UNet, ResNeXt101UNet
    # ConvNeXtSmallUNet, ConvNeXtBaseUNet, ConvNeXtTinyUNet, ConvNeXtLargeUNet
import celldetection as cd
from torchvision import transforms as trans
from scipy.ndimage import zoom
import torchvision.ops.boxes as bx
from . import functional as fun

__all__ = [
    'CPN',
    'CpnSlimU22', 'CpnU22', 'CpnWideU22',
    'CpnResNet50UNet', 'CpnResNet34UNet', 'CpnResNet18UNet', 'CpnResNet101UNet', 'CpnResNeXt50UNet', 'CpnResNet152UNet',
    # 'CpnResNet18FPN', 'CpnResNet34FPN', 'CpnResNet50FPN',
    # 'CpnResNet101FPN', 'CpnResNet152FPN', 'CpnResNeXt50FPN', 'CpnResNeXt101FPN', 'CpnResNeXt152FPN',
    # 'CpnWideResNet50FPN', 'CpnWideResNet101FPN',  # 'CpnMobileNetV3LargeFPN', 'CpnMobileNetV3SmallFPN',
    'CpnResUNet', 'CpnResNeXt101UNet',
    # 'CpnConvNeXtBaseUNet', 'CpnConvNeXtLargeUNet', 'CpnConvNeXtTinyUNet',
    # 'ConvNeXtSmallUNet'
]


# adapted from torchvision.ops.boxes
def remove_small_boxes(boxes: Tensor, min_size: float, return_mask=False) -> Tensor:
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    if return_mask:
        return keep
    keep = torch.where(keep)[0]
    return keep


class CPNCore(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_channels,
            order,
            samples: int,
            score_channels: int,
            refinement: bool = True,
            refinement_margin: float = 3.,
            uncertainty_head=False,
            contour_features='1',
            refinement_features='0',
            contour_head_channels=None,
            contour_head_stride=1,
            refinement_head_channels=None,
            refinement_head_stride=1,
            refinement_interpolation='bilinear',
            refinement_buckets=1,
    ):
        super().__init__()
        self.uncertainty_avg = False
        self.order = order
        self.backbone = backbone
        self.refinement_features = refinement_features
        self.contour_features = contour_features
        self.refinement_interpolation = refinement_interpolation
        assert refinement_buckets >= 1
        self.refinement_buckets = refinement_buckets

        encoder_out_channels = self.backbone.out_channels  # should match enc-out
        if isinstance(backbone_channels, int):
            contour_head_input_channels = refinement_head_input_channels = backbone_channels
            raise NotImplementedError
        elif isinstance(backbone_channels, (tuple, list)):
            contour_head_input_channels = backbone_channels[int(contour_features)]
            refinement_head_input_channels = backbone_channels[int(refinement_features)]
            score_head_input_channels = encoder_out_channels[int(contour_features)] + contour_head_input_channels
            refinement_head_input_channels_cat = refinement_head_input_channels + encoder_out_channels[
                int(refinement_features)]
        elif isinstance(backbone_channels, dict):
            contour_head_input_channels = backbone_channels[contour_features]
            refinement_head_input_channels = backbone_channels[refinement_features]
            score_head_input_channels = encoder_out_channels[contour_features] + contour_head_input_channels
            refinement_head_input_channels_cat = refinement_head_input_channels + encoder_out_channels[
                refinement_features]
        else:
            raise ValueError('Did not understand type of backbone_channels')
        self.score_reduce = nn.Conv2d(score_head_input_channels, contour_head_input_channels, 1, bias=False)
        self.score_head = ReadOut(
            contour_head_input_channels, score_channels,
            kernel_size=7,  # 3
            padding=3,
            channels_mid=contour_head_channels,
            stride=contour_head_stride
        )
        self.score_logsoft = nn.LogSoftmax(dim=1) if score_channels == 2 else nn.Identity()
        self.location_head = ReadOut(
            contour_head_input_channels, 2,
            kernel_size=7,
            padding=3,
            channels_mid=contour_head_channels,
            stride=contour_head_stride,
        )
        self.fourier_head = ReadOut(
            contour_head_input_channels, order * 4,
            kernel_size=7,
            padding=3,
            channels_mid=contour_head_channels,
            stride=contour_head_stride
        )
        self.uncertainty_head = ReadOut(
            contour_head_input_channels, 4,
            kernel_size=7,
            padding=3,
            channels_mid=contour_head_channels,
            stride=contour_head_stride,
            # activation='sigmoid'
        ) if uncertainty_head else None
        if refinement:
            self.refinement_reduce = nn.Conv2d(refinement_head_input_channels_cat, refinement_head_input_channels, 1,
                                               bias=False)
            self.refinement_head = ReadOut(
                refinement_head_input_channels, 2 * refinement_buckets,
                kernel_size=7,
                padding=3,
                final_activation=ScaledTanh(refinement_margin),
                channels_mid=refinement_head_channels,
                stride=refinement_head_stride
            )
            self.refinement_margin = 1.  # legacy
        else:
            self.refinement_head = None
            self.refinement_margin = None

    def head_dropout_(self, val, heads=None):
        heads = (self.score_head, self.fourier_head, self.location_head) if heads is None else heads
        for head in heads:
            for handle, key, mod in cd.iter_submodules(head, nn.Dropout2d):
                if val:
                    mod.train()
                else:
                    mod.eval()

    def forward(self, inputs):
        features, encoder_features = self.backbone(inputs)

        if isinstance(features, torch.Tensor):
            contour_features = refinement_features = features
            raise NotImplementedError
        else:
            contour_features = features[self.contour_features]
            refinement_features = features[self.refinement_features]
            enc_score_features = encoder_features[self.contour_features]
            enc_refinement_features = encoder_features[self.refinement_features]

        score_features = torch.cat((contour_features, enc_score_features), 1)
        score_features = self.score_reduce(score_features)
        scores = self.score_head(score_features)
        locations = self.location_head(contour_features)
        fourier = self.fourier_head(contour_features)

        refinement = None
        if self.refinement_head is not None:
            refinement_features = torch.cat((refinement_features, enc_refinement_features), 1)
            refinement_features = self.refinement_reduce(refinement_features)
            refinement = self.refinement_head(refinement_features) * self.refinement_margin
            if refinement.shape[-2:] != inputs.shape[-2:]:  # 337 ns
                # bilinear: 3.79 ms for (128, 128) to (512, 512)
                # bicubic: 11.5 ms for (128, 128) to (512, 512)
                refinement = F.interpolate(refinement, inputs.shape[-2:],
                                           mode=self.refinement_interpolation, align_corners=False)

        if self.uncertainty_head is None:
            uncertainty = None
        else:
            uncertainty = self.uncertainty_head(contour_features)

        return scores, locations, refinement, fourier, uncertainty


class CPN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            certainty_thresh: float = None,
            samples: int = 32,
            classes: int = 2,

            uncertainty_head: bool = False,
            iou_loss='giou',
            box_loss=True,
            binary_loss=False,
            threshold_loss=False,

            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            contour_features='1',
            refinement_features='0',

            contour_head_channels=None,
            contour_head_stride=1,
            order_weights=True,
            refinement_head_channels=None,
            refinement_head_stride=1,
            refinement_interpolation='bilinear',

            image_mean: List[float] = None,
            image_std: List[float] = None,

            uncertainty_factor=1.,
            uncertainty_nms=True,

            box_confidence=False,
    ):
        """CPN base class.

        This is the base class for the Contour Proposal Network.

        References:
            https://www.sciencedirect.com/science/article/pii/S136184152200024X

        Args:
            backbone: A backbone network. E.g. ``cd.models.U22(in_channels, 0)``.
            order: Contour order. The higher, the more complex contours can be proposed.
                ``order=1`` restricts the CPN to propose ellipses, ``order=3`` allows for non-convex rough outlines,
                ``order=8`` allows even finer detail.
            nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with
                ``iou > nms_thresh`` to be identical.
            score_thresh: Score threshold. For binary classification problems (object vs. background) an object must
                have ``score > score_thresh`` to be proposed as a result.
            samples: Number of samples. This sets the number of coordinates with which a contour is defined.
                This setting can be changed on the fly, e.g. small for training and large for inference.
                Small settings reduces computational costs, while larger settings capture more detail.
            classes: Number of classes. Default: 2 (object vs. background).
            refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of
                the proposed contours.
            refinement_iterations: Number of refinement iterations.
            refinement_margin: Maximum refinement margin (step size) per iteration.
            refinement_buckets: Number of refinement buckets. Bucketed refinement is especially recommended for data
                with overlapping objects. ``refinement_buckets=1`` practically disables bucketing,
                ``refinement_buckets=6`` uses 6 different buckets, each influencing different fractions of a contour.
            contour_features: If ``backbone`` returns a dictionary of features, this is the key used to retrieve
                the features that are used to predict contours.
            refinement_features: If ``backbone`` returns a dictionary of features, this is the key used to retrieve
                the features that are used to predict the refinement tensor.
            contour_head_channels: Number of intermediate channels in contour ``ReadOut`` Modules. By default, this is the
                number of incoming feature channels.
            contour_head_stride: Stride used for the contour prediction. Larger stride means less contours can
                be proposed in total, which speeds up execution times.
            order_weights: Whether to use order specific weights.
            refinement_head_channels: Number of intermediate channels in refinement ``ReadOut`` Modules. By default,
                this is the number of incoming feature channels.
            refinement_head_stride: Stride used for the refinement prediction. Larger stride means less detail, but
                speeds up execution times.
            refinement_interpolation: Interpolation mode that is used to ensure that refinement tensor and input
                image have the same shape.
        """
        super().__init__()
        self.order = order
        self.nms_thresh = nms_thresh
        self.samples = samples
        self.score_thresh = score_thresh
        self.certainty_thresh = certainty_thresh
        self.score_channels = classes
        self.refinement = refinement
        self.refinement_iterations = refinement_iterations
        self.refinement_margin = refinement_margin
        self.functional = False
        self.full_detail = False
        self.score_target_dtype = None
        self.iou_loss = iou_loss
        self.box_loss = box_loss
        self.binary_loss = binary_loss
        self.uncertainty_factor = uncertainty_factor
        self.uncertainty_nms = uncertainty_nms
        self.box_confidence = box_confidence

        self.teacher_score_thresh_fg = .9
        self.teacher_score_thresh_bg = .1
        self.teacher_uncertainty_margin = .1
        self.student_uncertainty_margin = .5

        if threshold_loss:
            raise NotImplementedError

        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        self.transform = trans.Compose([
            trans.Normalize(mean=image_mean, std=image_std)
        ])

        if not hasattr(backbone, 'out_channels'):
            raise ValueError('Backbone should have an attribute out_channels that states the channels of its output.')

        if iou_loss and samples < 64:
            warnings.warn('The iou_loss option of the CPN is enabled, but the `samples` setting is rather low. '
                          'This may impair detection performance. Increase `samples` or provide box targets manually.')

        self.core = CPNCore(
            backbone=backbone,
            backbone_channels=backbone.out_channels,
            order=order,
            samples=samples,
            uncertainty_head=uncertainty_head,
            score_channels=classes,
            refinement=refinement,
            refinement_margin=refinement_margin,
            contour_features=contour_features,
            refinement_features=refinement_features,
            contour_head_channels=contour_head_channels,
            contour_head_stride=contour_head_stride,
            refinement_head_channels=refinement_head_channels,
            refinement_head_stride=refinement_head_stride,
            refinement_interpolation=refinement_interpolation,
            refinement_buckets=refinement_buckets,
        )

        self.order_weights = 1.
        if isinstance(order_weights, bool):
            if order_weights:
                self.order_weights = nn.Parameter(order_weighting(self.order), requires_grad=False)
        else:
            self.order_weights = order_weights

        self.objectives = OrderedDict({
            'score': nn.CrossEntropyLoss(),
            'fourier': nn.L1Loss(reduction='none'),
            'location': nn.L1Loss(),
            'contour': nn.L1Loss(),
            'confidence': nn.L1Loss(),
            'refinement': nn.L1Loss(),
            'boxes': nn.L1Loss(),
            'binary': None,  # TverskyLoss(),
        })
        self.weights = {
            'fourier': 1.,  # note: fourier has order specific weights
            'location': 1.,
            'contour': 1.5,  # 3
            'score_fg': 1.,
            'score_bg': 2.,
            'refinement': 1.,
            'boxes': .88,
            'iou': 1.,
            'uncertainty': 1.,
            'binary': 1.,

            'ssod_boxes': .88,
            'ssod_score': 1.,
        }

        self._rel_location2abs_location_cache: Dict[str, Tensor] = {}
        self._fourier2contour_cache: Dict[str, Tensor] = {}

    def compute_loss(
            self,
            uncertainty,
            fourier,
            locations,
            contours,
            refined_contours,
            boxes,
            raw_scores,
            targets: dict,
            labels,
            fg_masks,
            b
    ):
        raise NotImplementedError  # not needed in docker

    def process_scores(self, scores):
        if self.score_channels == 1:
            scores = torch.sigmoid(scores)
            classes = torch.squeeze((scores > self.score_thresh).long(), 1)
        elif self.score_channels == 2:
            scores = F.softmax(scores, dim=1)[:, 1:2]
            classes = torch.squeeze((scores > self.score_thresh).long(), 1)
        elif self.score_channels > 2:
            scores = F.softmax(scores, dim=1)
            classes = torch.argmax(scores, dim=1).long()
        else:
            raise ValueError
        return scores, classes

    def forward(
            self,
            inputs,
            targets: Dict[str, Tensor] = None,
            nms=True
    ):
        # Presets
        original_size = inputs.shape[-2:]
        assert torch.all((inputs >= 0.) & (inputs <= 1.)), 'Inputs should be in interval 0..1'
        inputs = self.transform(inputs)

        # Core
        scores, locations, refinement, fourier, uncertainty = self.core(inputs)
        participation_loss = scores[:1].mean() + locations[:1].mean() + refinement[:1].mean() + fourier[:1].mean()
        if uncertainty is not None:
            participation_loss = participation_loss + uncertainty[:1].mean()
        participation_loss = participation_loss * 0.

        # Scores
        raw_scores = scores
        scores, classes = self.process_scores(scores)

        actual_size = fourier.shape[-2:]
        n, c, h, w = fourier.shape
        if self.functional:
            fourier = fourier.view((n, c // 2, 2, h, w))
        else:
            fourier = fourier.view((n, c // 4, 4, h, w))

        # Maybe apply changed order
        if self.order < self.core.order:
            fourier = fourier[:, :self.order]

        # Fetch sampling and labels
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            sampling = targets.get('sampling')
            labels = targets['labels']
        else:
            sampling = None
            labels = classes.detach()
        labels = downsample_labels(labels[:, None], actual_size)[:, 0]

        # Locations
        # raw_locations = locations.detach()
        locations = rel_location2abs_location(locations, cache=self._rel_location2abs_location_cache)

        # Extract proposals
        fg_mask = labels > 0
        if self.certainty_thresh is not None and uncertainty is not None:
            fg_mask &= uncertainty.sigmoid().mean(1) < (1 - self.certainty_thresh)

        b, y, x = torch.where(fg_mask)
        selected_fourier = fourier[b, :, :, y, x]  # Tensor[-1, order, 4]
        selected_locations = locations[b, :, y, x]  # Tensor[-1, 2]
        selected_classes = classes[b, y, x]

        if self.score_channels in (1, 2):
            selected_scores = scores[b, 0, y, x]  # Tensor[-1]
        elif self.score_channels > 2:
            selected_scores = scores[b, selected_classes, y, x]  # Tensor[-1]
        else:
            raise ValueError
        score_uncertainty = None

        if uncertainty is None:
            selected_uncertainties = None
        else:
            selected_uncertainties = uncertainty[b, :, y, x]

        if sampling is not None:
            sampling = sampling[b]

        # Convert to pixel space
        selected_contour_proposals, sampling = fouriers2contours(selected_fourier, selected_locations,
                                                                 samples=self.samples, sampling=sampling,
                                                                 cache=self._fourier2contour_cache)

        contour_uncertainty = None  # removed option

        # Rescale in case of multi-scale
        selected_contour_proposals = scale_contours(actual_size=actual_size, original_size=original_size,
                                                    contours=selected_contour_proposals)
        selected_fourier, selected_locations = scale_fourier(actual_size=actual_size, original_size=original_size,
                                                             fourier=selected_fourier, location=selected_locations)

        if self.refinement and self.refinement_iterations > 0:
            det_indices = selected_contour_proposals  # Tensor[num_contours, samples, 2]
            num_loops = self.refinement_iterations
            if self.training and num_loops > 1:
                num_loops = torch.randint(low=1, high=num_loops + 1, size=())

            for _ in torch.arange(0, num_loops):
                det_indices = torch.round(det_indices.detach())
                det_indices[..., 0].clamp_(0, original_size[1] - 1)
                det_indices[..., 1].clamp_(0, original_size[0] - 1)
                indices = det_indices.detach().long()  # Tensor[-1, samples, 2]
                if self.core.refinement_buckets == 1:
                    responses = refinement[b[:, None], :, indices[:, :, 1], indices[:, :, 0]]  # Tensor[-1, samples, 2]
                else:
                    buckets = resolve_refinement_buckets(sampling, self.core.refinement_buckets)
                    responses = None
                    for bucket_indices, bucket_weights in buckets:
                        bckt_idx = torch.stack((bucket_indices * 2, bucket_indices * 2 + 1), -1)
                        cur_ref = refinement[b[:, None, None], bckt_idx, indices[:, :, 1, None], indices[:, :, 0, None]]
                        cur_ref = cur_ref * bucket_weights[..., None]
                        if responses is None:
                            responses = cur_ref
                        else:
                            responses = responses + cur_ref
                det_indices = det_indices + responses
            selected_contours = det_indices
        else:
            selected_contours = selected_contour_proposals
        selected_contours[..., 0].clamp_(0, original_size[1] - 1)
        selected_contours[..., 1].clamp_(0, original_size[0] - 1)

        # Bounding boxes
        if selected_contours.numel() > 0:
            selected_boxes = cd.ops.contours2boxes(selected_contours, axis=1)  # 43.3 µs ±290 ns for Tensor[2203, 32, 2]
        else:
            selected_boxes = torch.empty((0, 4), device=selected_contours.device)

        # Loss
        if self.training:
            loss, losses, info = self.compute_loss(
                uncertainty=selected_uncertainties,
                fourier=selected_fourier,
                locations=selected_locations,
                contours=selected_contour_proposals,
                refined_contours=selected_contours,
                boxes=selected_boxes,
                raw_scores=raw_scores,
                targets=targets,
                labels=labels,
                fg_masks=fg_mask,
                b=b
            )
            loss = loss + participation_loss * 0.
        else:
            loss = losses = info = None

        if self.training and not self.full_detail:
            return OrderedDict({
                'loss': loss,
                'losses': losses,
                'info': info,
            })

        final_contours = []
        final_boxes = []
        final_scores = []
        final_classes = []
        final_locations = []
        final_fourier = []
        final_contour_proposals = []
        final_selected_score_uncertainties = []
        final_selected_contour_uncertainties = []
        final_selected_uncertainties = []
        for batch_index in range(inputs.shape[0]):
            sel = b == batch_index
            final_contours.append(selected_contours[sel])
            final_boxes.append(selected_boxes[sel])
            final_scores.append(selected_scores[sel])
            final_classes.append(selected_classes[sel])
            final_locations.append(selected_locations[sel])
            final_fourier.append(selected_fourier[sel])
            final_contour_proposals.append(selected_contour_proposals[sel])
            if contour_uncertainty is not None:
                final_selected_contour_uncertainties.append(contour_uncertainty[sel])
            if score_uncertainty is not None:
                final_selected_score_uncertainties.append(score_uncertainty[sel])
            if selected_uncertainties is not None:
                final_selected_uncertainties.append(selected_uncertainties[sel])

        if not self.training and nms:
            if self.uncertainty_nms and len(final_selected_uncertainties):
                nms_weights = [s * (1. - u.sigmoid().mean(1)) for s, u in
                               zip(final_scores, final_selected_uncertainties)]
                assert len(nms_weights) == len(final_scores)
            else:
                nms_weights = final_scores
            keeps = batched_box_nmsi(final_boxes, nms_weights, iou_threshold=self.nms_thresh)
            for j in range(len(final_boxes)):
                final_boxes[j] = final_boxes[j][keeps[j]]
                final_scores[j] = final_scores[j][keeps[j]]
                final_contours[j] = final_contours[j][keeps[j]]
                final_locations[j] = final_locations[j][keeps[j]]
                final_fourier[j] = final_fourier[j][keeps[j]]
                final_contour_proposals[j] = final_contour_proposals[j][keeps[j]]
                final_classes[j] = final_classes[j][keeps[j]]
                if len(final_selected_score_uncertainties):
                    final_selected_score_uncertainties[j] = final_selected_score_uncertainties[j][keeps[j]]
                if len(final_selected_contour_uncertainties):
                    final_selected_contour_uncertainties[j] = final_selected_contour_uncertainties[j][keeps[j]]
                if len(final_selected_uncertainties):
                    final_selected_uncertainties[j] = final_selected_uncertainties[j][keeps[j]]

        # The dict below can be altered to return additional items of interest
        outputs = OrderedDict({
            'box_uncertainties': final_selected_uncertainties,
            'score_uncertainties': final_selected_score_uncertainties,
            'contour_uncertainties': final_selected_contour_uncertainties,
            'contours': final_contours,
            'boxes': final_boxes,
            'scores': final_scores,
            'classes': final_classes,
            'loss': loss,
            'losses': losses,
            'info': info,
            'score_maps': scores,
        })

        return outputs


def _make_cpn_doc(title, text, backbone):
    return f"""{title}

    {text}

    References:
        https://www.sciencedirect.com/science/article/pii/S136184152200024X

    Args:
        in_channels: Number of input channels.
        order: Contour order. The higher, the more complex contours can be proposed.
            ``order=1`` restricts the CPN to propose ellipses, ``order=3`` allows for non-convex rough outlines,
            ``order=8`` allows even finer detail.
        nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with
            ``iou > nms_thresh`` to be identical.
        score_thresh: Score threshold. For binary classification problems (object vs. background) an object must
            have ``score > score_thresh`` to be proposed as a result.
        samples: Number of samples. This sets the number of coordinates with which a contour is defined.
            This setting can be changed on the fly, e.g. small for training and large for inference.
            Small settings reduces computational costs, while larger settings capture more detail.
        classes: Number of classes. Default: 2 (object vs. background).
        refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of
            the proposed contours.
        refinement_iterations: Number of refinement iterations.
        refinement_margin: Maximum refinement margin (step size) per iteration.
        refinement_buckets: Number of refinement buckets. Bucketed refinement is especially recommended for data
            with overlapping objects. ``refinement_buckets=1`` practically disables bucketing,
            ``refinement_buckets=6`` uses 6 different buckets, each influencing different fractions of a contour.
        backbone_kwargs: Additional backbone keyword arguments. See docstring of ``{backbone}``.
        **kwargs: Additional CPN keyword arguments. See docstring of ``cd.models.CPN``.

    """


class CpnU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=U22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with U-Net 22 backbone.',
        'A Contour Proposal Network that uses a U-Net with 22 convolutions as a backbone.',
        'cd.models.U22'
    )


class CpnResUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResUNet(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Residual U-Net backbone.',
        'A Contour Proposal Network that uses a U-Net build with residual blocks.',
        'cd.models.ResUNet'
    )


class CpnSlimU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=SlimU22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Slim U-Net 22 backbone.',
        'A Contour Proposal Network that uses a Slim U-Net as a backbone. '
        'Slim U-Net has 22 convolutions with less feature channels than normal U22.',
        'cd.models.SlimU22'
    )


class CpnWideU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideU22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Wide U-Net 22 backbone.',
        'A Contour Proposal Network that uses a Wide U-Net as a backbone. '
        'Wide U-Net has 22 convolutions with more feature channels than normal U22.',
        'cd.models.WideU22'
    )


class CpnResNet101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 101 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 101 U-Net as a backbone.',
        'cd.models.ResNet101UNet'
    )


class CpnResNet152UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet152UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 152 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 152 U-Net as a backbone.',
        'cd.models.ResNet152UNet'
    )


class CpnResNeXt101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 101 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNeXt 101 U-Net as a backbone.',
        'cd.models.ResNeXt101UNet'
    )


class CpnResNeXt50UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt50UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 50 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNeXt 50 U-Net as a backbone.',
        'cd.models.ResNeXt50UNet'
    )


class CpnResNet50UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        encoder = ResNet50UNet(in_channels, 0, **backbone_kwargs)
        super().__init__(
            backbone=encoder,
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 50 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 50 U-Net as a backbone.',
        'cd.models.ResNet50UNet'
    )


class CpnResNet34UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet34UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 34 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 34 U-Net as a backbone.',
        'cd.models.ResNet34UNet'
    )


class CpnResNet18UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .5,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet18UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 18 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 18 U-Net as a backbone.',
        'cd.models.ResNet18UNet'
    )

#
# class CpnResNet18FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNet18FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNet 18 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNet 18 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNet18FPN'
#     )
#
#
# class CpnResNet34FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNet34FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNet 34 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNet 34 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNet34FPN'
#     )
#
#
# class CpnResNet50FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNet50FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNet 50 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNet 50 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNet50FPN'
#     )
#
#
# class CpnResNet101FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNet101FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNet 101 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNet 101 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNet101FPN'
#     )
#
#
# class CpnResNet152FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNet152FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNet 152 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNet 152 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNet152FPN'
#     )
#
#
# class CpnResNeXt50FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNeXt50FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNeXt 50 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNeXt 50 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNeXt50FPN'
#     )
#
#
# class CpnResNeXt101FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNeXt101FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNeXt 101 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNeXt 101 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNeXt101FPN'
#     )
#
#
# class CpnResNeXt152FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=ResNeXt152FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ResNeXt 152 FPN backbone.',
#         'A Contour Proposal Network that uses a ResNeXt 152 Feature Pyramid Network as a backbone.',
#         'cd.models.ResNeXt152FPN'
#     )
#
#
# class CpnWideResNet50FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=WideResNet50FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with Wide ResNet 50 FPN backbone.',
#         'A Contour Proposal Network that uses a Wide ResNet 50 Feature Pyramid Network as a backbone.',
#         'cd.models.WideResNet50FPN'
#     )
#
#
# class CpnWideResNet101FPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=WideResNet101FPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with Wide ResNet 101 FPN backbone.',
#         'A Contour Proposal Network that uses a Wide ResNet 101 Feature Pyramid Network as a backbone.',
#         'cd.models.WideResNet101FPN'
#     )


# class CpnMobileNetV3SmallFPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=MobileNetV3SmallFPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with Small MobileNetV3 FPN backbone.',
#         'A Contour Proposal Network that uses a Small MobileNetV3 Feature Pyramid Network as a backbone.',
#         'cd.models.MobileNetV3SmallFPN'
#     )
#
#
# class CpnMobileNetV3LargeFPN(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs: dict = None,
#             **kwargs
#     ):
#         super().__init__(
#             backbone=MobileNetV3LargeFPN(in_channels, **(backbone_kwargs or {})),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with Large MobileNetV3 FPN backbone.',
#         'A Contour Proposal Network that uses a Large MobileNetV3 Feature Pyramid Network as a backbone.',
#         'cd.models.MobileNetV3LargeFPN'
#     )


# class CpnConvNeXtSmallUNet(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs=None,
#             **kwargs
#     ):
#         backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
#         super().__init__(
#             backbone=ConvNeXtSmallUNet(in_channels, 0, **backbone_kwargs),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ConvNeXt Small U-Net backbone.',
#         'A Contour Proposal Network that uses a ConvNeXt Small U-Net as a backbone.',
#         'cd.models.ConvNeXtSmallUNet'
#     )
#
#
# class CpnConvNeXtTinyUNet(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs=None,
#             **kwargs
#     ):
#         backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
#         super().__init__(
#             backbone=ConvNeXtTinyUNet(in_channels, 0, **backbone_kwargs),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ConvNeXt Tiny U-Net backbone.',
#         'A Contour Proposal Network that uses a ConvNeXt Tiny U-Net as a backbone.',
#         'cd.models.ConvNeXtTinyUNet'
#     )
#
#
# class CpnConvNeXtLargeUNet(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs=None,
#             **kwargs
#     ):
#         backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
#         super().__init__(
#             backbone=ConvNeXtLargeUNet(in_channels, 0, **backbone_kwargs),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ConvNeXt Large U-Net backbone.',
#         'A Contour Proposal Network that uses a ConvNeXt Large U-Net as a backbone.',
#         'cd.models.ConvNeXtLargeUNet'
#     )
#
#
# class CpnConvNeXtBaseUNet(CPN):
#     def __init__(
#             self,
#             in_channels: int,
#             order: int = 5,
#             nms_thresh: float = .2,
#             score_thresh: float = .5,
#             samples: int = 32,
#             classes: int = 2,
#             refinement: bool = True,
#             refinement_iterations: int = 4,
#             refinement_margin: float = 3.,
#             refinement_buckets: int = 1,
#             backbone_kwargs=None,
#             **kwargs
#     ):
#         backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
#         super().__init__(
#             backbone=ConvNeXtBaseUNet(in_channels, 0, **backbone_kwargs),
#             order=order,
#             nms_thresh=nms_thresh,
#             score_thresh=score_thresh,
#             samples=samples,
#             classes=classes,
#             refinement=refinement,
#             refinement_iterations=refinement_iterations,
#             refinement_margin=refinement_margin,
#             refinement_buckets=refinement_buckets,
#             **kwargs
#         )
#
#     __init__.__doc__ = _make_cpn_doc(
#         'Contour Proposal Network with ConvNeXt Base U-Net backbone.',
#         'A Contour Proposal Network that uses a ConvNeXt Base U-Net as a backbone.',
#         'cd.models.ConvNeXtBaseUNet'
#     )


models_by_name = {
    'cpn_u22': 'cpn_u22'
}


def get_cpn(name):
    return fetch_model(models_by_name[name])

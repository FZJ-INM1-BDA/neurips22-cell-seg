from functools import partial
import celldetection as cd
import torch.nn as nn
import torch
import os
from os.path import basename, dirname, join
from . import cpn_model as cpn
from .losses import TverskyLoss, FocalTverskyLoss, LossComposition
from celldetection import lookup_nn
import inspect

__all__ = ['build_cpn_model']


class NormProxy:  # from newer version of celldetection
    def __init__(self, norm, **kwargs):
        self.norm = norm
        self.kwargs = kwargs

    def __call__(self, num_channels):
        Norm = lookup_nn(self.norm, call=False)
        kwargs = dict(self.kwargs)
        args = inspect.getfullargspec(Norm).args
        if 'num_features' in args:
            kwargs['num_features'] = num_channels
        elif 'num_channels' in args:
            kwargs['num_channels'] = num_channels
        return Norm(**kwargs)

    def __repr__(self):
        return f'NormProxy({self.norm}, kwargs={self.kwargs})'

    __str__ = __repr__


def _build_model(conf):
    backbone_kwargs = dict(conf.get('backbone_kwargs', {}))
    backbone_kwargs['backbone_kwargs'] = dict(conf.get('encoder_kwargs', {}))
    decoder_block = conf.get('decoder_block')
    block_cls = None
    if decoder_block is not None:
        norm_layer = NormProxy('GroupNorm', num_groups=32) if conf.get('decoder_group_norm') else None
        downsample = partial(cd.models.ConvNorm, norm_layer=norm_layer) if conf.get('decoder_group_norm') else None
        block_cls = partial(getattr(cd.models, decoder_block), activation=conf.get('decoder_activation'),
                            norm_layer=norm_layer, downsample=downsample)
        backbone_kwargs['block_cls'] = block_cls

    print("Build model")
    Model = getattr(cpn, conf.get('model', conf.get('cpn')))
    assert Model is not None, 'Set either conf.model or conf.cpn'
    model = Model(
        in_channels=conf.in_channels,
        order=conf.order,
        nms_thresh=conf.nms_thresh,
        score_thresh=conf.score_thresh,
        certainty_thresh=conf.get('certainty_thresh', None),
        samples=conf.samples,
        classes=conf.classes,
        iou_loss=conf.get('loss_iou', 'giou'),
        binary_loss=conf.get('loss_binary', False),
        threshold_loss=conf.get('loss_threshold', False),
        refinement=conf.get('refinement', True),
        refinement_iterations=conf.get('refinement_iterations', 4),
        refinement_margin=conf.get('refinement_margin', 3),
        refinement_buckets=conf.get('refinement_buckets', 6),
        contour_features=conf.get('contour_features', '1'),
        refinement_features=conf.get('refinement_features', '0'),
        contour_head_stride=conf.get('contour_head_stride', 1),
        order_weights=conf.get('order_weights', True),
        refinement_head_stride=conf.get('refinement_head_stride', 1),
        refinement_interpolation=conf.get('refinement_interpolation', 'bilinear'),
        # backbone_kwargs={} if block_cls is None else dict(block_cls=block_cls),
        image_std=conf.get('image_std'),
        image_mean=conf.get('image_mean'),
        uncertainty_head=conf.get('uncertainty_head', False),
        uncertainty_factor=conf.get('uncertainty_factor', 1.),
        box_confidence=conf.get('box_confidence', True),
        backbone_kwargs=backbone_kwargs,
        uncertainty_nms=conf.get('uncertainty_nms', False),
    )

    print('Tweak model')
    if conf.get('tweaks') is not None:
        cd.conf2tweaks_(conf.tweaks, model)
    return model


def resolve_class_objective(conf, patch_conf):
    lcs = conf.loss_classification.split(',')
    objectives = []
    for lc in lcs:
        if lc == 'ce':  # or (conf.classes > 1 and patch_conf)
            assert conf.classes > 1
            # conf.loss_classification = 'ce'
            objectives.append(nn.CrossEntropyLoss(label_smoothing=conf.get('label_smoothing', .1)))
        elif lc == 'bce':
            objectives.append(nn.BCEWithLogitsLoss())
        elif lc == 'focal':
            objectives.append(cd.models.SigmoidFocalLoss(alpha=conf.loss_focal_alpha,
                                                         gamma=conf.loss_focal_gamma))
        elif lc == 'soft-focal':
            objectives.append(cd.models.SigmoidSoftFocalLoss(alpha=conf.loss_focal_alpha,
                                                             gamma=conf.loss_focal_gamma))
        elif lc == 'tversky':
            objectives.append(TverskyLoss(conf.get('loss_tversky_alpha', .5), conf.get('loss_tversky_beta', .5)))
        elif lc == 'focal-tversky':
            objectives.append(FocalTverskyLoss(conf.get('loss_tversky_alpha', .5), conf.get('loss_tversky_beta', .5),
                                               conf.get('loss_tversky_gamma', 1.5)))
        else:
            raise ValueError(lcs)
    if len(objectives) == 1:
        return objectives[0]
    else:
        return LossComposition(*objectives)


def build_cpn_model(conf, verbose=True, patch_conf=True, patch_model=True, patch_unc=None):
    checkpoint = conf.get('checkpoint')
    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            if verbose:
                print('Load checkpoint')
            model_ = torch.load(checkpoint, map_location='cpu')
            model_conf = cd.Config.from_json(join(dirname(checkpoint), 'config.json'))
            model: cpn.CPN = _build_model(model_conf)
            model.load_state_dict(model_.state_dict(), strict=False)
            if patch_conf:
                conf.model = model_conf.get('model', model_conf.get('cpn', conf.model))
                conf.classes = model_conf.classes
                conf.uncertainty_head = model_conf.uncertainty_head
                conf.order = model_conf.order
                # [...]
        else:
            if verbose:
                print('Fetch model')
            model = cd.fetch_model(checkpoint, map_location='cpu')
    elif conf.model in dir(cpn):
        model = _build_model(conf)

        if verbose:
            print('Init decoder')
        if 'unet' in str(conf.model).lower():
            for m in model.core.backbone.unet.modules():
                if isinstance(m, nn.Conv2d):
                    if verbose:
                        print('Init decoder', m, flush=True)
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out',
                        nonlinearity=conf.decoder_activation.lower().replace('leakyrelu', 'leaky_relu')
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    if verbose:
                        print('Init decoder', m, flush=True)
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    else:
        raise ValueError

    if verbose:
        print('Update classification objective')
    model.objectives['score'] = resolve_class_objective(conf, patch_conf)

    if verbose:
        print('Update regression objective')
    if conf.loss_fourier == 'l1':
        model.objectives['fourier'] = nn.L1Loss(reduction='none')
    else:
        raise NotImplementedError(conf.loss_fourier)

    if conf.loss_regression == 'l1':
        model.objectives['location'] = nn.L1Loss()
        model.objectives['contour'] = nn.L1Loss()
        model.objectives['refinement'] = nn.L1Loss()
        model.objectives['boxes'] = nn.L1Loss()
    elif conf.loss_regression == 'mse':
        model.objectives['location'] = nn.MSELoss()
        model.objectives['contour'] = nn.MSELoss()
        model.objectives['refinement'] = nn.MSELoss()
        model.objectives['boxes'] = nn.MSELoss()
    elif conf.loss_regression == 'smooth-l1':
        model.objectives['location'] = nn.SmoothL1Loss(beta=conf.loss_regression_beta)
        model.objectives['contour'] = nn.SmoothL1Loss(beta=conf.loss_regression_beta)
        model.objectives['refinement'] = nn.SmoothL1Loss(beta=conf.loss_regression_beta)
        model.objectives['boxes'] = nn.SmoothL1Loss(beta=conf.loss_regression_beta)
    # [...]
    else:
        raise NotImplementedError(conf.loss_regression)

    return model

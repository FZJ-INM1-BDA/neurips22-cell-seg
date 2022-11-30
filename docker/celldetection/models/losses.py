from torch.nn.modules.loss import _Loss
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss
from ..ops.losses import _margin_distance_loss, log_margin_loss, _single_margin_distance_loss, \
    _two_margin_distance_loss, margin_loss, soft_focal_loss, sigmoid_soft_focal_loss

__all__ = ['MarginLoss', 'LogMarginLoss', 'MarginDistanceLoss', 'SigmoidFocalLoss', 'SigmoidSoftFocalLoss',
           'SoftFocalLoss']


class MarginLoss(_Loss):
    def __init__(self, margin_pos=.9, margin_neg=None, exponent=2, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.m_pos = margin_pos
        self.m_neg = margin_neg
        self.exponent = exponent

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return margin_loss(input, target, reduction=self.reduction, m_neg=self.m_neg, m_pos=self.m_pos,
                           exponent=self.exponent)


class LogMarginLoss(_Loss):
    def __init__(self, margin_pos=.9, margin_neg=None, exponent=1, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.m_pos = margin_pos
        self.m_neg = margin_neg
        self.exponent = exponent

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return log_margin_loss(input, target, reduction=self.reduction, m_neg=self.m_neg, m_pos=self.m_pos,
                               exponent=self.exponent)


class MarginDistanceLoss(_Loss):
    def __init__(self, margin=.1, exponent=2, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        """Margin Distance Loss.

        Args:
            margin: Allowed margin or tuple of margins as `(margin_for_target_0, margin_for_target_1)`.
                Both `margin=(.1, .1)` and `margin=.1` allow a distance of `0.1` to the respective targets.
            exponent:
            size_average:
            reduce:
            reduction:
        """
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.exponent = exponent
        self.fn = _two_margin_distance_loss if isinstance(margin, (tuple, list)) else _single_margin_distance_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _margin_distance_loss(input, target, self.margin, self.exponent, self.reduction, self.fn)


class _FocalLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', alpha=.5, gamma=2) -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.gamma = gamma


class SigmoidFocalLoss(_FocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma,
                                  reduction=self.reduction)


class SigmoidSoftFocalLoss(_FocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return sigmoid_soft_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)


class SoftFocalLoss(_FocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return soft_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)

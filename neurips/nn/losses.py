import torch.nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from .functional.losses import tversky_loss, focal_tversky_loss

__all__ = ['TverskyLoss', 'FocalTverskyLoss', 'LossComposition']


class TverskyLoss(_Loss):
    def __init__(self, alpha=.5, beta=.5, size_average=None, reduce=None, reduction: str = 'mean',
                 sigmoid=True) -> None:
        if reduction != 'mean':
            raise NotImplementedError(reduction)
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.beta = beta
        self.sigmoid = sigmoid

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return tversky_loss(input, target, alpha=self.alpha, beta=self.beta, sigmoid=self.sigmoid)


class FocalTverskyLoss(_Loss):
    def __init__(self, alpha=.5, beta=.5, gamma=1.5, size_average=None, reduce=None, reduction: str = 'mean',
                 sigmoid=True) -> None:
        if reduction != 'mean':
            raise NotImplementedError(reduction)
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigmoid = sigmoid

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return focal_tversky_loss(input, target, alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                                  sigmoid=self.sigmoid)


class LossComposition(torch.nn.Module):
    def __init__(self, *objectives):
        super().__init__()
        self.objectives = objectives

    def forward(self, inputs, targets):
        losses = []
        for obj in self.objectives:
            losses.append(obj(inputs, targets))
        return sum(losses) / len(losses)

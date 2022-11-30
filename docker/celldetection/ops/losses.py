import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ['reduce_loss', 'log_margin_loss', 'margin_loss', 'r1_regularization', 'soft_focal_loss',
           'sigmoid_soft_focal_loss']


def reduce_loss(x: Tensor, reduction: str, **kwargs):
    """Reduce loss.

    Reduces Tensor according to ``reduction``.

    Args:
        x: Input.
        reduction: Reduction method. Must be a symbol of ``torch``.
        **kwargs: Additional keyword arguments.

    Returns:
        Reduced Tensor.
    """
    if reduction == 'none':
        return x
    fn = getattr(torch, reduction, None)
    if fn is None:
        raise ValueError(f'Unknown reduction: {reduction}')
    return fn(x, **kwargs)


def log_margin_loss(inputs: Tensor, targets: Tensor, m_pos=.9, m_neg=None, exponent=1, reduction='mean', eps=1e-6):
    if m_neg is None:
        m_neg = 1 - m_pos

    pos = torch.pow(F.relu_(torch.log(m_pos / (inputs + eps))), exponent)
    neg = torch.pow(F.relu_(torch.log((1 - m_neg) / (1 - inputs + eps))), exponent)
    loss = targets * pos + (1 - targets) * neg
    return reduce_loss(loss, reduction)


def margin_loss(inputs: Tensor, targets: Tensor, m_pos=.9, m_neg=None, exponent=2, reduction='mean'):
    if m_neg is None:
        m_neg = 1 - m_pos

    pos = torch.pow(F.relu_(m_pos - inputs), exponent)
    neg = torch.pow(F.relu_(inputs - m_neg), exponent)
    loss = targets * pos + (1 - targets) * neg
    return reduce_loss(loss, reduction)


def _single_margin_distance_loss(inputs, targets, diff, m):
    return F.relu(diff - m)


def _two_margin_distance_loss(inputs, targets, diff, margin: tuple):
    m_neg, m_pos = margin
    m = (inputs > targets).int()
    return F.relu(m * (diff - m_pos)) + F.relu((1 - m) * (diff - m_neg))


def _margin_distance_loss(inputs, targets, margin, exponent, reduction, fn):
    diff = F.l1_loss(inputs, targets, reduction='none')
    v = fn(inputs, targets, diff, margin)
    return reduce_loss(torch.pow(v, exponent), reduction)


def margin_distance_loss(inputs: Tensor, targets: Tensor, margin=.1, exponent=2, reduction='mean'):
    fn = _two_margin_distance_loss if isinstance(margin, (tuple, list)) else _single_margin_distance_loss
    return _margin_distance_loss(inputs, targets, margin=margin, exponent=exponent, reduction=reduction, fn=fn)


def r1_regularization(logits, inputs, gamma=1., reduction='sum'):
    r"""R1 regularization.

    A gradient penalty regularization.
    This regularization may for example be applied to a discriminator with real data:

    .. math::

        R_1(\psi) &:= \frac{\gamma}{2} \mathbb E_{ p_{\mathcal D}(x)} \left[\|\nabla D_\psi(x)\|^2\right]

    References:
        - https://arxiv.org/pdf/1801.04406.pdf (Eq. 9)
        - https://arxiv.org/pdf/1705.09367.pdf
        - https://arxiv.org/pdf/1711.09404.pdf

    Examples:
        >>> real.requires_grad_(True)
        ... real_logits = discriminator(real)
        ... loss_d_real = F.softplus(-real_logits)
        ... loss_d_r1 = r1_regularization(real_logits, real)
        ... loss_d_real = (loss_d_r1 + loss_d_real).mean()
        ... loss_d_real.backward()
        ... real.requires_grad_(False)

    Args:
        logits: Logits.
        inputs: Inputs.
        gamma: Gamma.
        reduction: How to reduce all non-batch dimensions. E.g. ``'sum'`` or ``'mean'``.

    Returns:
        Penalty Tensor[n].
    """
    grads = torch.autograd.grad(logits.sum(), inputs=inputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = reduce_loss(grads.square(), reduction, dim=list(range(1, grads.ndim)))
    return penalty * (gamma * .5)


def _soft_focal_loss(
        with_logits: bool,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        gamma: float,
        reduction: str,
):
    if with_logits:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

    p_t = p * targets + (1 - p) * (1 - targets)
    factor = p_t * ((1 - p_t) ** gamma - 1) + 1
    loss = ce_loss * factor

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_soft_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.5,
        gamma: float = 2,
        reduction: str = 'mean',
):
    return _soft_focal_loss(
        with_logits=True,
        inputs=inputs,
        targets=targets,
        alpha=alpha,
        gamma=gamma,
        reduction=reduction
    )


def soft_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.5,
        gamma: float = 2,
        reduction: str = 'mean',
):
    return _soft_focal_loss(
        with_logits=True,
        inputs=inputs,
        targets=targets,
        alpha=alpha,
        gamma=gamma,
        reduction=reduction
    )

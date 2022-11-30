import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import warnings
from bisect import bisect_right
from collections import Counter

__all__ = ['WarmupMultiStepLR']


def get_warmup_factor(step, steps=1000, factor=0.001, method='linear'):
    if step >= steps:
        return 1.
    if method == 'constant':
        return factor
    elif method == 'linear':
        a = step / steps
        return factor * (1 - a) + a
    raise ValueError(f'Unknown method: {method}')


class WarmupMultiStepLR(lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            warmup_factor=0.001,
            warmup_steps=1000,
            warmup_method='linear',
            gamma=.1,
            last_epoch=-1,
            verbose=False
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))
        factor = get_warmup_factor(self.last_epoch, self.warmup_steps, self.warmup_factor, self.warmup_method)
        return [base_lr * factor * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]

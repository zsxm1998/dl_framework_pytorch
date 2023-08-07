import math
import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmUpLR(_LRScheduler):
    def __init__(self, optimizer, T_total, warm_up_lr=1e-8, warm_up_step=0, eta_min=0, last_epoch=-1, verbose=False):
        self.T_total = T_total
        self.warm_up_lr = warm_up_lr #min(warm_up_lr + self.base_lr)
        self.warm_up_step = warm_up_step
        self.T_cos = T_total - warm_up_step
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmUpLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        epoch = self.last_epoch % self.T_total
        if epoch < self.warm_up_step:
            return [self.warm_up_lr + epoch *
                    (base_lr - self.warm_up_lr) / self.warm_up_step
                    for base_lr in self.base_lrs]
        elif epoch == self.warm_up_step:
            return self.base_lrs
        return [(1 + math.cos(math.pi * (epoch-self.warm_up_step) / self.T_cos)) /
                (1 + math.cos(math.pi * (epoch-self.warm_up_step-1) / self.T_cos)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        epoch = self.last_epoch % self.T_total
        if epoch < self.warm_up_step:
            return [self.warm_up_lr + epoch *
                    (base_lr - self.warm_up_lr) / self.warm_up_step
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (epoch-self.warm_up_step) / self.T_cos)) / 2
                    for base_lr in self.base_lrs]
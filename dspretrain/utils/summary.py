""" Summary utilities
Modification of timm's code.

Title: pytorch-image-models
Author: Ross Wightman
Date: 2021
Availability: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/summary.py
"""

import csv
from collections import OrderedDict
try: 
    import wandb  # type: ignore
except ImportError:
    pass

def original_update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    if train_metrics:
        rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

class AverageStdMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.sum_sq += (val * val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def std(self):
        mean = self.avg
        var = self.sum_sq / self.count - mean * mean
        return max(var, 0.0) ** 0.5
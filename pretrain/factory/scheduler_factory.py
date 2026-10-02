"""Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""

from typing import List, Optional, Union

from torch.optim import Optimizer

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler


def step_scheduler_kwargs(cfg):
    """cfg/argparse to kwargs helper
    Convert scheduler args in argparse args or cfg (.dot) like object to keyword args.
    """

    kwargs = dict(
        sched=cfg.sched,
        decay_iters=getattr(cfg, "decay_iters", 300),
        decay_milestones=getattr(cfg, "decay_milestones", [300, 600]),
        warmup_iters=getattr(cfg, "warmup_iters", 50),
        cooldown_iters=getattr(cfg, "cooldown_iters", 0),
        decay_rate=getattr(cfg, "decay_rate", 0.1),
        min_lr=getattr(cfg, "min_lr", 0.0),
        warmup_lr=getattr(cfg, "warmup_lr", 1e-5),
        noise=getattr(cfg, "lr_noise", None),
        noise_pct=getattr(cfg, "lr_noise_pct", 0.67),
        noise_std=getattr(cfg, "lr_noise_std", 1.0),
        noise_seed=getattr(cfg, "seed", 42),
        cycle_mul=getattr(cfg, "lr_cycle_mul", 1.0),
        cycle_decay=getattr(cfg, "lr_cycle_decay", 0.1),
        cycle_limit=getattr(cfg, "lr_cycle_limit", 1),
        k_decay=getattr(cfg, "lr_k_decay", 1.0),
    )
    return kwargs


def create_step_scheduler(
    optimizer: Optimizer,
    total_updates: int = 1000,
    sched: str = "cosine",
    decay_iters: int = 300,
    decay_milestones: List[int] = (300, 600),
    cooldown_iters: int = 0,
    decay_rate: float = 0.1,
    min_lr: float = 0,
    warmup_lr: float = 1e-5,
    warmup_iters: int = 0,
    noise: Union[float, List[float]] = None,
    noise_pct: float = 0.67,
    noise_std: float = 1.0,
    noise_seed: int = 42,
    cycle_mul: float = 1.0,
    cycle_decay: float = 0.1,
    cycle_limit: int = 1,
    k_decay: float = 1.0,
):
    t_initial = total_updates
    warmup_t = warmup_iters
    decay_t = decay_iters
    cooldown_t = cooldown_iters

    # warmup args
    warmup_args = dict(
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_t,
        warmup_prefix=False,  # Warmup prefix is always False with this setup (warmup is included inside the number of training iterations)
    )

    # setup noise args for supporting schedulers
    if noise is not None:
        if isinstance(noise, (list, tuple)):
            noise_range = [n * t_initial for n in noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = noise * t_initial
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=noise_pct,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    # setup cycle args for supporting schedulers
    cycle_args = dict(
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
    )

    lr_scheduler = None
    if sched == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=False,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == "tanh":
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=False,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    elif sched == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_rate,
            t_in_epochs=False,
            **warmup_args,
            **noise_args,
        )
    elif sched == "multistep":
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_milestones,
            decay_rate=decay_rate,
            t_in_epochs=False,
            **warmup_args,
            **noise_args,
        )
    elif sched == "poly":
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=False,
            k_decay=k_decay,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )

    return lr_scheduler

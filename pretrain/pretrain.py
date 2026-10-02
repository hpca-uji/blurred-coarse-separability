#!/usr/bin/env python3
"""

Real-Time VisualAtoms pre-train script based on training script provided by timm

Simplified version with changes:
- No apex
- No AugMix
- Forced PyTorch RandAugment in GPU instead of timm's as a dataset transform
- Custom CollateFunction to apply PyTorch's MixUp/CutMixPyTorch + label_smoothing
- VisualAtoms dataset generation on GPU
- Forced BCE/SoftTargetCrossEntropy loss function
- The concept of epoch dissapears as unique images are generated in Real-Time
- Scheduler called on step instead of epoch
- Saver called on step
- No input_size or in_channels parameters since generated images are always 3channels
- One dataloader worker per GPU training process
- Forced syncbatchnorm when ditributed training and no splitbn when augment repetition
"""

import argparse
import copy
import importlib
import logging
import os
import time
from contextlib import suppress
from datetime import datetime
from functools import partial

import yaml
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
)

from torch.nn.parallel import DistributedDataParallel
from timm import utils
from timm.layers import convert_sync_batchnorm, set_fast_norm
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy
from timm.models import (
    model_parameters,
    create_model,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import NativeScaler

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, "compile")

from factory.augmentation_factory import VatomPrefetchLoader
from factory.dataset_factory import create_vatom_dataset
from factory.scheduler_factory import step_scheduler_kwargs, create_step_scheduler
from factory.saver_factory import UpdateSaver, resume_checkpoint
from factory.utils import AverageStdMeter

_logger = logging.getLogger("train")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="Real-Time VisualAtoms Pre-Training")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")
# Keep this argument outside the dataset group because it is positional.
group.add_argument(
    "--dataset-csv-path",
    type=str,
    help="Path to VisualAtom-1k original classes parameters",
)
group.add_argument(
    "--dataset-cfg-path",
    type=str,
    help="Path to classes configuration file",
)
group.add_argument(
    "--dataset-cfg-select",
    type=str,
    help="Configuration to select from file",
    default="DEFAULT",
)
group.add_argument(
    "--dataset-size",
    type=int,
    help="Number of instances to generate",
    default=100000000,
)
group.add_argument(
    "--res", type=int, help="Image resolution of generated images", default=224
)
parser.add_argument(
    "--kernel-file",
    type=str,
    required=True,
    help="Path to kernel batch code",
)
group.add_argument(
    "--kernel-res", type=int, help="Image resolution of generated images", default=512
)
parser.add_argument(
    "--min-rand-kernel-res",
    type=int,
    default=0,
    help="Minimum kernel resolution when random resolution",
    )
# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model",
    default="resnet50",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "resnet50")',
)
group.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
group.add_argument(
    "--pretrained-path",
    default=None,
    type=str,
    help="Load this checkpoint as if they were the pretrained weights (with adaptation).",
)
group.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Load this checkpoint into model after initialization (default: none)",
)
group.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
group.add_argument(
    "--num-classes",
    type=int,
    default=21000,
    metavar="N",
    help="number of label classes (Model default if None)",
)
group.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
group.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="Input batch size for training (default: 128)",
)
group.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
group.add_argument(
    "--fuser",
    default="",
    type=str,
    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')",
)
group.add_argument(
    "--grad-accum-steps",
    type=int,
    default=1,
    metavar="N",
    help="The number of steps to accumulate gradients (default: 1)",
)
group.add_argument(
    "--grad-checkpointing",
    action="store_true",
    default=False,
    help="Enable gradient checkpointing through model blocks/stages",
)
group.add_argument(
    "--fast-norm",
    default=False,
    action="store_true",
    help="enable experimental fast-norm",
)
group.add_argument("--model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument(
    "--head-init-scale", default=None, type=float, help="Head initialization scale"
)
group.add_argument(
    "--head-init-bias", default=None, type=float, help="Head initialization bias value"
)
group.add_argument(
    "--torchcompile-mode",
    type=str,
    default=None,
    help="torch.compile mode (default: None).",
)

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="torch.jit.script the full model",
)
scripting_group.add_argument(
    "--torchcompile",
    nargs="?",
    type=str,
    default=None,
    const="inductor",
    help="Enable compilation w/ specified backend (default: inductor).",
)

# Device & distributed
group = parser.add_argument_group("Device parameters")
group.add_argument(
    "--device", default="cuda", type=str, help="Device (accelerator) to use."
)
group.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="Use PyTorch AMP for mixed precision training",
)
group.add_argument(
    "--amp-dtype",
    default="float16",
    type=str,
    help="lower precision AMP dtype (default: float16)",
)
group.add_argument(
    "--model-dtype",
    default=None,
    type=str,
    help="Model dtype override (non-AMP) (default: float32)",
)
group.add_argument(
    "--no-ddp-bb",
    action="store_true",
    default=False,
    help="Force broadcast buffers for native DDP to off.",
)
group.add_argument(
    "--synchronize-step",
    action="store_true",
    default=False,
    help="torch.cuda.synchronize() end of each step",
)
group.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--device-modules",
    default=None,
    type=str,
    nargs="+",
    help="Python imports for device backend modules.",
)

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument(
    "--opt",
    default="adamw",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "adamw")',
)
group.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
group.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
group.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
group.add_argument(
    "--weight-decay", type=float, default=2e-5, help="weight decay (default: 2e-5)"
)
group.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
group.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)
group.add_argument(
    "--layer-decay",
    type=float,
    default=None,
    help="layer-wise learning rate decay (default: None)",
)
group.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

# Loss parameters
group = parser.add_argument_group("Trainig loss parameters")
group.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
group.add_argument(
    "--bce-sum",
    action="store_true",
    default=False,
    help="Sum over classes when using BCE loss.",
)
group.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled).",
)
group.add_argument(
    "--bce-pos-weight",
    type=float,
    default=None,
    help="Positive weighting for BCE loss.",
)

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument(
    "--sched",
    type=str,
    default="cosine",
    metavar="SCHEDULER",
    help='LR scheduler (default: "cosine")',
)
group.add_argument(
    "--lr",
    type=float,
    default=1.0e-3,
    metavar="LR",
    help="learning rate, overrides lr-base if set (default: None)",
)
group.add_argument(
    "--lr-base",
    type=float,
    default=0.1,
    metavar="LR",
    help="base learning rate: lr = lr_base * global_batch_size / base_size",
)
group.add_argument(
    "--lr-base-size",
    type=int,
    default=256,
    metavar="DIV",
    help="base learning rate batch size (divisor, default: 256).",
)
group.add_argument(
    "--lr-base-scale",
    type=str,
    default="",
    metavar="SCALE",
    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)',
)
group.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
group.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
group.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-decay",
    type=float,
    default=0.5,
    metavar="MULT",
    help="amount to decay each learning rate cycle (default: 0.5)",
)
group.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit, cycles enabled if > 1",
)
group.add_argument(
    "--lr-k-decay",
    type=float,
    default=1.0,
    help="learning rate k-decay for cosine/poly (default: 1.0)",
)
group.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-6,
    metavar="LR",
    help="warmup learning rate (default: 1e-5)",
)
group.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (default: 0)",
)
group.add_argument(
    "--decay-milestones",
    default=[300, 600],
    type=int,
    nargs="+",
    metavar="MILESTONES",
    help="list of decay iter indices for multistep lr. must be increasing",
)
group.add_argument(
    "--decay-iters",
    type=float,
    default=42000,
    metavar="N",
    help="iter interval to decay LR",
)
group.add_argument(
    "--warmup-iters",
    type=int,
    default=5000,
    metavar="N",
    help="iterrs to warmup LR, if scheduler supports",
)
group.add_argument(
    "--cooldown-iters",
    type=int,
    default=0,
    metavar="N",
    help="iters to cooldown LR at min_lr, after cyclic schedule ends",
)
group.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.5,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
parser.add_argument(
    "--prep",
    type=str,
    required=True,
    choices=["crop", "resize", "none"],
    help="Whether to apply random crop, resize or none during data preprocessing",
)
group.add_argument(
    "--num-ops", type=int, help="Number of RandomAugment layers", default=0
)
group.add_argument(
    "--magnitude", type=int, help="Magnitude to apply on RandomAugment", default=28
)
group.add_argument(
    "--aug-repeats",
    type=int,
    default=0,
    help="Number of augmentation repetitions (distributed training only) (default: 0)",
)
group.add_argument(
    "--reprob",
    type=float,
    default=0.25,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
group.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
group.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
group.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
group.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
group.add_argument(
    "--drop-path",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
group.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group(
    "Batch norm parameters", "Only works with gen_efficientnet based models currently."
)
group.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
group.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)

# Model Exponential Moving Average
group = parser.add_argument_group("Model exponential moving average parameters")
group.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights.",
)
group.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
group.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="Decay factor for model weights moving average (default: 0.9998)",
)
group.add_argument(
    "--model-ema-warmup", action="store_true", help="Enable warmup for model EMA decay."
)

# Misc
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
group.add_argument(
    "--worker-seeding", type=str, default="all", help="worker seed mode (default: all)"
)
group.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
group.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to save (default: 10)",
)
group.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input batches every log interval for debugging",
)
group.add_argument(
    "--profile",
    action="store_true",
    default=False,
    help="Activate profiler and cut training",
)
group.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
group.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_profiler():
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=900, warmup=90, active=10),
        record_shapes=False,
        with_stack=False,
    )


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)

    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ("float32", "float16", "bfloat16")
        model_dtype = getattr(torch, args.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning(
                "float16 is not recommended for training, for half precision bfloat16 is recommended."
            )

    # resolve AMP arguments
    use_amp = None
    if args.amp:
        assert (
            model_dtype is None or model_dtype == torch.float32
        ), "float32 model dtype must be used with AMP"
        assert args.amp_dtype in ("float16", "bfloat16")
        use_amp = "native"
        if args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs["pretrained_cfg_overlay"] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=3,  # Model will always process 3-channels images
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    )
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    # move model to GPU, enable channels last layout if set
    model.to(
        device=device, dtype=model_dtype
    )  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed:
        model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not args.torchcompile
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = (
                "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
            )
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
            )

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults["weight_decay"] = (
            args.weight_decay
        )  # this isn't stored in optimizer.defaults
        defaults = ", ".join([f"{k}: {v}" for k, v in defaults.items()])
        logging.info(
            f"Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}"
        )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "native":
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type in ("cuda",) and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            _logger.info(
                f"AMP not enabled. Training in {model_dtype or torch.float32}."
            )

    # Default values
    init_datapoints = 0
    dataset_size = args.dataset_size
    max_history = args.checkpoint_hist
    current_checkpoint = 1
    if args.resume:
        # Load resume checkpoint
        resume_vars = resume_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=utils.is_primary(args),
        )
        if resume_vars:  # If different than None, overload vars
            dataset_size, init_datapoints, current_checkpoint, max_history = resume_vars
            current_checkpoint += 1  # Add one to update var for next checkpoint

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # setup distributed training
    if args.distributed:
        if utils.is_primary(args):
            _logger.info("Using native Torch DistributedDataParallel.")
        model = DistributedDataParallel(
            model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb
        )
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert (
            has_compile
        ), "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        model = torch.compile(
            model, backend=args.torchcompile, mode=args.torchcompile_mode
        )

    # Create vatom dataset
    dataset_train, classes, norbits_max, nvertex_max = create_vatom_dataset(
        dataset_csv_path=args.dataset_csv_path,
        dataset_cfg_path=args.dataset_cfg_path,
        dataset_cfg_select=args.dataset_cfg_select,
        init_datapoints=init_datapoints,
        batch_size=args.batch_size,
        nclasses=args.num_classes,
        aug_repeats=args.aug_repeats,
        device=args.rank,
        gpus=args.world_size,
    )

    # Create dataloader with forced 1 worker
    loader_base = DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    # Create dataloader
    loader_train = VatomPrefetchLoader(
        loader=loader_base,
        kernel_file=args.kernel_file,
        device=args.rank,
        batch_size=args.batch_size,
        classes=classes,
        nclasses=args.num_classes,
        res=args.res,
        kernel_res=args.kernel_res,
        min_rand_kernel_res=args.min_rand_kernel_res,
        norbits_max=norbits_max,
        nvertex_max=nvertex_max,
        prep=args.prep,
        num_ops=args.num_ops,
        magnitude=args.magnitude,
        mixup=args.mixup,
        cutmix=args.cutmix,
        label_smoothing=args.smoothing,
        reprob=args.reprob,
    )

    # Compute global batch size
    global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
    # Total updates of the current type of model
    total_updates = dataset_size * args.aug_repeats / global_batch_size
    done_updates = int(init_datapoints * args.aug_repeats / global_batch_size)
    # Compute updates to complete current type of model
    pending_updates = int(
        (args.dataset_size - init_datapoints) * args.aug_repeats / global_batch_size
    )

    if args.bce_loss:
        train_loss_fn = BinaryCrossEntropy(
            target_threshold=args.bce_target_thresh,
            sum_classes=args.bce_sum,
            pos_weight=args.bce_pos_weight,
        )
    else:
        train_loss_fn = SoftTargetCrossEntropy()

    # setup checkpoint saver
    saver = None
    output_dir = None
    if utils.is_primary(args):

        # Create folder save path
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                ]
            )
        output_dir = utils.get_outdir(
            args.output if args.output else "./output/train", exp_name
        )

        # Create ckp saver
        saver = UpdateSaver(
            model=model,
            optimizer=optimizer,
            dataset_size=dataset_size,
            global_batch_size=global_batch_size,
            total_updates=total_updates,
            aug_repeats=args.aug_repeats,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            save_prefix=output_dir,
            current_checkpoint=current_checkpoint,
            max_history=max_history,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    # setup scheduler
    lr_scheduler = create_step_scheduler(
        optimizer=optimizer,
        total_updates=total_updates,
        **step_scheduler_kwargs(args),
    )
    # Update scheduler
    lr_scheduler.step_update(done_updates)

    # Set up profiler
    prof = None
    if args.profile:

        # Create profiler
        if utils.is_primary(args):
            prof = get_profiler()

    try:

        final_loss = train_loaderfull(
            model,
            loader_train,
            optimizer,
            prof,
            done_updates,
            pending_updates,
            train_loss_fn,
            args,
            device=device,
            lr_scheduler=lr_scheduler,
            saver=saver,
            output_dir=output_dir,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
        )

    except KeyboardInterrupt:
        pass

    # Log final loss of the model
    if utils.is_primary(args):
        # Log info
        _logger.info(f"Training ended; Final loss: {final_loss}  ")

    # Clear distributed process group
    if args.distributed:
        dist.destroy_process_group()


def train_loaderfull(
    model,
    loader,
    optimizer,
    prof,
    done_updates,
    pending_updates,
    loss_fn,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
):

    if prof:
        prof.start()

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = AverageStdMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    accum_counter = 0
    num_updates = done_updates
    optimizer.zero_grad()
    update_sample_count = 0

    data_start_time = update_start_time = time.time()

    for step, (input, target) in enumerate(loader):

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        stall_time = time.time() - data_start_time

        # Break early if profiling
        if args.profile and step == 1000:
            loader.clear_ctx()
            break

        if num_updates >= pending_updates:
            loader.clear_ctx()
            break

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(
                        model, exclude_head="agc" in args.clip_mode
                    ),
                    create_graph=second_order,
                    need_update=accum_counter + 1 == accum_steps,
                )
            else:
                _loss.backward(create_graph=second_order)
                if accum_counter + 1 == accum_steps:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(
                                model, exclude_head="agc" in args.clip_mode
                            ),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and accum_counter + 1 < accum_steps:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        # Perform optimizer update when accumulation is complete
        if accum_counter + 1 == accum_steps:
            num_updates += 1
            accum_counter = 0
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model, step=num_updates)

            if args.synchronize_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "npu":
                    torch.npu.synchronize()

            time_now = time.time()
            time_update = time_now - update_start_time
            update_time_m.update(time_update)
            data_time_m.update(stall_time / time_update * 100)
            update_start_time = time_now

            if utils.is_primary(args):
                saver.step_save(num_updates=num_updates)

            if num_updates % args.log_interval == 0:
                lrl = [param_group["lr"] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                loss_avg, loss_now = losses_m.avg, losses_m.val
                if args.distributed:
                    loss_avg = utils.reduce_tensor(
                        loss.new([loss_avg]), args.world_size
                    ).item()
                    loss_now = utils.reduce_tensor(
                        loss.new([loss_now]), args.world_size
                    ).item()
                    update_sample_count *= args.world_size

                if utils.is_primary(args):
                    _logger.info(
                        f"Train: [{num_updates:>4d}/{pending_updates} "
                        f"({100. * num_updates / pending_updates:>3.0f}%)]  "
                        f"Loss: {loss_now:#.3g} ({loss_avg:#.3g})  "
                        f"Time: {update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s  "
                        f"LR: {lr:.3e}  "
                        f"Stall: {data_time_m.avg:.3f}%, {data_time_m.std:.3f}%"
                    )

                    if args.save_images and output_dir:
                        save_image(
                            input,
                            os.path.join(output_dir, f"train-batch-{num_updates}.jpg"),
                            padding=0,
                            normalize=True,
                        )

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            update_sample_count = 0
        else:
            accum_counter += 1

        # Call profiler step if it exists
        if prof:
            prof.step()

        data_start_time = time.time()

    # Save profiler trace if it exists
    if prof:
        prof.stop()
        prof.export_chrome_trace(f"trace.json")

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()

    del input
    del target

    return loss_avg


if __name__ == "__main__":
    main()

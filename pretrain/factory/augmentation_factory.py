from typing import Any
from pathlib import Path

import numpy as np
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.auto_aug import rand_augment
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIInterpType
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    MixUp,
    CutMix,
    RandomChoice,
    Normalize,
    ToDtype,
)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
import pycuda.driver as drv
from pycuda.compiler import SourceModule

_FILL = 128

## Create paths to necessary include dirs
ROOT = Path(__file__).resolve().parent.parent.parent
CUDACORE = f"{ROOT}/src/cuda/core"
SINGLEKERNELS = f"{ROOT}/src/cuda/single/kernels"


class Holder(drv.PointerHolderBase):
    """Class to use torch Tensors on cuda kernel"""

    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


@pipeline_def(enable_conditionals=True, num_threads=2, prefetch_queue_depth=1)
def rapipe(prep, n, m, res):

    # Input
    imgs = fn.external_source(name="input", device="gpu", layout="HWC")

    # Random crop resize
    if prep == "crop":
        imgs = fn.random_resized_crop(
            imgs,
            size=(res, res),
            device="gpu",
            interp_type=DALIInterpType.INTERP_CUBIC,
            antialias=True,
        )
    elif prep == "resize":
        # Normal resizing
        imgs = fn.resize(
            imgs,
            size=(res, res),
            device="gpu",
            interp_type=DALIInterpType.INTERP_CUBIC,
            antialias=True,
        )

    # Apply rand augment
    imgs = rand_augment.rand_augment(data=imgs, shape=(res, res), n=n, m=m)

    return imgs


class VatomPrefetchLoader:

    def __init__(
        self,
        loader: DataLoader,
        kernel_file: str,
        device: int,
        batch_size: int,
        classes: list,
        res: int = 224,
        kernel_res: int = 512,
        min_rand_kernel_res: int = 0,
        nclasses: int = 1000,
        norbits_max: int = 200,
        nvertex_max: int = 1000,
        prep: str = "crop",
        num_ops: int = 2,
        magnitude: int = 9,
        mixup: float = 0.5,
        cutmix: float = 0.5,
        label_smoothing: float = 0.1,
        reprob: float = 0.25,
    ):
        self.loader = loader
        self.device = device
        self.batch_size = batch_size
        self.res = res
        self.kernel_res = kernel_res
        self.min_rand_kernel_res = min_rand_kernel_res
        self.classes = classes
        self.nclasses = nclasses
        self.label_smoothing = label_smoothing
        self.norbits_max = norbits_max
        self.totalvertex_max = norbits_max * nvertex_max

        # --- Initialize pycuda and push context for this device (one-time) ---
        drv.init()
        self.ctx = drv.Device(device).retain_primary_context()
        self.ctx.push()

        # Load code
        with open(kernel_file, "r") as f:
            kernel_code = f.read()

        # --- compile module from string once and store function handle ---
        self.mod = SourceModule(
            source=kernel_code,
            options=["--maxrregcount=64"],
            include_dirs=[CUDACORE, SINGLEKERNELS],
            no_extern_c=True,
        )

        self.genvatom = self.mod.get_function(f"genvatom_simple_batch")

        # --- allocate per-prefetcher reusable device buffers (persistent) ---
        # Adjust types/sizes to match your kernel expectations
        self.colors = torch.empty(
            [self.batch_size * norbits_max], dtype=torch.uint8, device=self.device
        )
        self.vertex = torch.empty(
            [self.batch_size * self.totalvertex_max, 2],
            dtype=torch.float64,
            device=self.device,
        )
        self.noise = torch.empty(
            [self.batch_size * self.totalvertex_max, 2],
            dtype=torch.float32,
            device=self.device,
        )
        self.pos = torch.empty(
            [self.batch_size, 2], dtype=torch.float64, device=self.device
        )
        self.prepare_classes_pointers()

        self.rapipe = rapipe(
            prep=prep,
            n=num_ops,
            m=magnitude,
            res=res,
            device_id=device,
            batch_size=batch_size,
        )
        self.rapipe.build()
        self.todtype = ToDtype(torch.float32, scale=True)

        # Define mixup/cutmix
        self.mix = None

        # Set up mixup
        if mixup > 0.0:
            mixup_fn = MixUp(alpha=mixup, num_classes=nclasses)
            self.mix = mixup_fn

        # Set up cutmix
        if cutmix > 0.0:
            cutmix_fn = CutMix(alpha=cutmix, num_classes=nclasses)
            self.mix = cutmix_fn

        # If both active, create RandomChoice
        if mixup > 0.0 and cutmix > 0.0:

            self.mix = RandomChoice([mixup_fn, cutmix_fn])

        # Create norm layer
        self.norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        if reprob > 0:
            self.re = RandomErasing(probability=reprob, mode="pixel", device=device)
        else:
            self.re = None

    def prepare_classes_pointers(self):

        for i in range(len(self.classes)):
            self.classes[i] = Holder(self.classes[i])

    def clear_ctx(self):
        self.ctx.pop()
        self.ctx.detach()

    def __iter__(self):
        """
        Underlying loader must yield batches of (idxs_cpu_tensor, class_ids_cpu_tensor), both on CPU.
        Example collate: collate idxs/class ids into 1D CPU tensors with shape (B,)
        """

        first = True

        # Create stream
        pycuda_stream = drv.Stream()
        torch_stream = torch.cuda.ExternalStream(pycuda_stream.handle)

        for idxs, labels in self.loader:

            # Enqueue kernel launches + aug on generation stream
            with torch.cuda.stream(torch_stream):

                # Convert idxs to scalar
                idx_scalar = idxs.item()

                # Move labels to device (non_blocking) - they might be converted to soft labels by MixUp
                labels = labels[0].to(self.device, non_blocking=True)
                labels_kernel = labels.int()

                # preallocate device batch output buffer once per batch
                # kernel writes a single-channel uint8 image per sample; we keep (B,1,H,W)
                if self.min_rand_kernel_res == 0:
                    gen_res = self.kernel_res
                else:  # Radom resolution
                    g = torch.Generator().manual_seed(idx_scalar)
                    gen_res = torch.randint(
                        low=self.min_rand_kernel_res, high=1024, size=(1,), generator=g
                    ).item()
                out = torch.zeros(
                    (self.batch_size, gen_res, gen_res, 1),
                    dtype=torch.uint8,
                    device=self.device,
                )

                # Convert idxs to scalar
                idxs = idxs.item()

                # Create kernel args
                args = [
                    Holder(out),
                    Holder(self.vertex),
                    Holder(self.colors),
                    Holder(self.pos),
                    np.int32(self.totalvertex_max),
                    np.int32(self.norbits_max),
                    *self.classes,
                    np.int32(gen_res),
                    np.int32(idx_scalar),
                    np.int32(self.batch_size),
                    Holder(labels_kernel),
                ]

                # Launch kernel (grid/block values are placeholders—replace with your kernel's choices)
                self.genvatom(
                    *args,
                    grid=(self.batch_size, 1, 1),
                    block=(512, 1, 1),
                    stream=pycuda_stream,
                )

                # Expand to 3 channels
                out = out.repeat(1, 1, 1, 3)

                # Run and convert to tensor
                out = self.rapipe.run(input=out)
                out = out[0].as_tensor()
                out = torch.from_dlpack(out)
                out = out.permute(0, 3, 1, 2).contiguous()
                out = self.todtype(out)  # Batched todtype

                # Apply mixup
                if self.mix is not None:

                    out, labels = self.mix(out, labels)

                    labels = (
                        labels * (1.0 - self.label_smoothing)
                        + self.label_smoothing / self.nclasses
                    )
                else:
                    labels = torch.nn.functional.one_hot(labels, self.nclasses)

                # Apply batched norm
                out = self.norm(out)

                # Apply randomerasing
                if self.re:
                    out = self.re(out)

            # Yield one batch behind (prefetch pattern)
            if not first:
                yield prev_images, prev_labels
            else:
                first = False

            # Ensure main stream waits for generation stream so the returned tensors are ready
            torch.cuda.current_stream(device=self.device).wait_stream(torch_stream)

            # Save prepared batch for next yield
            prev_images = out
            prev_labels = labels

        # yield last prepared batch (if any)
        yield prev_images, prev_labels

    def __len__(self):
        return len(self.loader)

from typing import Tuple
from pathlib import Path

# General external imports
import pandas as pd
import numpy as np
from configparser import SectionProxy

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import ToDtype, Normalize

# PyCUDA imports
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# DALI imports
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIInterpType

# timm imports
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

## Create paths to necessary include dirs
ROOT = Path(__file__).resolve().parent.parent.parent
CUDACORE = f"{ROOT}/src/cuda/core"
SINGLEKERNELS = f"{ROOT}/src/cuda/single/kernels"


# Define Holder class to pass torch tensors to pycuda kernels
class Holder(drv.PointerHolderBase):
    """Class to use torch Tensors on cuda kernel"""

    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


## Basic dataset for vatoms
class VatomDataset(Dataset):
    def __init__(
        self,
        init_datapoints: int,
        nclasses: int,
    ):
        self.init_datapoints = init_datapoints
        self.nclasses = nclasses

    def __len__(self):
        # Length of the dataset is equal to number of classes
        return self.nclasses

    def __getitem__(self, idx):
        # Take init_datapoints into account
        return self.init_datapoints + idx, idx


# Build DALI pipeline
@pipeline_def(num_threads=2, prefetch_queue_depth=1)
def rapipe(res, prep):

    # Input
    imgs = fn.external_source(name="input", device="gpu", layout="HWC")

    if prep == "crop":
        # Random crop resize
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

    return imgs


class VatomBatchLoader:

    def __init__(
        self,
        loader: DataLoader,
        kernel_file: str,  # Path of the kernel code to execute
        device: torch.device,  # Device to generate data into
        samples_per_class: int,  # samples_per_class = batch_size
        classes: list,  # Lis of torch Tensors with class data
        norbits_max: int = 200,  # Maximim number of orbits of the generated VisualAtoms
        nvertex_max: int = 1000,  # Maximum nomber of vertex per orbit of the generated VisualAtoms
        res: int = 224,  # Resolution output of generated images
        prep: str = "crop",  # Whether to apply random crop, resize or none during data preprocessing
        kernel_res: int = 512,  # Kernel resolution
        min_rand_kernel_res: int = 0 # Minimum kernel resolution when random resolution 
    ):
        self.loader = loader
        self.device = device
        self.samples_per_class = samples_per_class
        self.classes = classes
        self.res = res
        self.kernel_res = kernel_res
        self.min_rand_kernel_res = min_rand_kernel_res
        self.norbits_max = norbits_max
        self.totalvertex_max = norbits_max * nvertex_max

        # --- Initialize pycuda and push context for this device (one-time) ---
        drv.init()
        self.ctx = drv.Device(device.index).retain_primary_context()
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
        self.colors = torch.empty(
            [self.samples_per_class * norbits_max], dtype=torch.uint8, device=self.device
        )
        self.vertex = torch.empty(
            [self.samples_per_class * self.totalvertex_max, 2],
            dtype=torch.float64,
            device=self.device,
        )
        self.pos = torch.empty(
            [self.samples_per_class, 2], dtype=torch.float64, device=self.device
        )

        # Apply Holder struct to data list of data tensors with class data
        self.prepare_classes_pointers()

        # Create pipeline
        self.rapipe = rapipe(
            res=res, prep=prep, device_id=device.index, batch_size=samples_per_class
        )
        self.rapipe.build()
        self.todtype = ToDtype(torch.float32, scale=True)

        # Create norm
        self.norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def prepare_classes_pointers(self):

        holders = []
        for i in range(len(self.classes)):
            holders.append(Holder(self.classes[i]))
        self.classes = holders

    def clear_ctx(self):
        self.ctx.pop()
        self.ctx.detach()

    def __iter__(self):

        first = True

        # Create stream
        pycuda_stream = drv.Stream()
        torch_stream = torch.cuda.ExternalStream(pycuda_stream.handle)

        for idx, label in self.loader:

            # Enqueue kernel launches + aug on generation stream
            with torch.cuda.stream(torch_stream):

                # Convert idxs to scalar
                idx_scalar = idx.item()

                # Create labels for kernel
                labels = torch.tensor([label[0]] * self.samples_per_class, dtype=torch.int32)
                # Move to device
                labels = labels.to(self.device, non_blocking=True)

                # preallocate device batch output buffer once per batch
                # kernel writes on a single-channel uint8 image per sample; we keep (B,1,H,W)
                # H,W depend on kernel_res
                if self.min_rand_kernel_res == 0:
                    gen_res = self.kernel_res
                else:  # Radom resolution
                    g = torch.Generator().manual_seed(idx_scalar)
                    gen_res = torch.randint(
                        low=self.min_rand_kernel_res, high=1024, size=(1,), generator=g
                    ).item()

                out = torch.zeros(
                    (self.samples_per_class, gen_res, gen_res, 1),
                    dtype=torch.uint8,
                    device=self.device,
                )

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
                    np.int32(self.samples_per_class),
                    Holder(labels),
                ]

                # Launch kernel (grid/block values are placeholders—replace with your kernel's choices)
                self.genvatom(
                    *args,
                    grid=(self.samples_per_class, 1, 1),
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

                # Apply batched norm if flagged
                if self.norm:
                    out = self.norm(out)

            # Yield one batch behind (prefetch pattern)
            if not first:
                yield prev_images
            else:
                first = False

            # Ensure main stream waits for generation stream so the returned tensors are ready
            torch.cuda.current_stream(device=self.device).wait_stream(torch_stream)

            # Save prepared batch for next yield
            prev_images = out

        # yield last prepared batch (if any)
        yield prev_images

    def __len__(self):
        return len(self.loader)


## Function that creates loader with specific precision backend and class parameters
def create_vatom_loader(
    kernel_file,
    device,
    samples_per_class,
    nclasses,
    classes,
    nvertex_max,
    norbits_max,
    res,
    prep,
    kernel_res,
    min_rand_kernel_res
):

    # Create dataset
    dataset = VatomDataset(init_datapoints=0, nclasses=nclasses)

    # Create loader
    loader = DataLoader(
        dataset,
        shuffle=False,  # Without shuffling to know order 1 to nclasses is garanteed
        batch_size=1,  # Only yield one pair number, class per iteration. Prefetcher will handle real batch_size=samples_per_class
        num_workers=1,  # Single worker since it is easy task but want it in different process than training
    )

    # Create VatomBatch Loader
    batch_loader = VatomBatchLoader(
        loader=loader,
        kernel_file=kernel_file,
        device=device,
        samples_per_class=samples_per_class,
        classes=classes,
        nvertex_max=nvertex_max,
        norbits_max=norbits_max,
        res=res,
        prep=prep,
        kernel_res=kernel_res,
        min_rand_kernel_res=min_rand_kernel_res
    )

    return batch_loader


def read_classes(
    classes_df: pd.DataFrame, device: torch.device
) -> Tuple[list[torch.Tensor], int, int]:

    # Convert to tensor

    # ---- Vertex ----
    vertex_tensor = torch.tensor(
        classes_df["Vertex"].values, dtype=torch.int32, device=device
    )
    nvertex_max = torch.max(vertex_tensor).cpu().item()

    # ---- Orbits ----
    norbits_tensor = torch.tensor(
        classes_df["Line_num"].values, dtype=torch.int32, device=device
    )
    norbits_max = torch.max(norbits_tensor).cpu().item()

    # ---- Oval rates ----
    ovalx_tensor = torch.tensor(
        classes_df["Oval_rate_x"].values, dtype=torch.float64, device=device
    )
    ovaly_tensor = torch.tensor(
        classes_df["Oval_rate_y"].values, dtype=torch.float64, device=device
    )

    # ---- Frequences ----
    freq1_tensor = torch.tensor(
        classes_df["nami 1"].values, dtype=torch.int32, device=device
    )
    freq2_tensor = torch.tensor(
        classes_df["nami 2"].values, dtype=torch.int32, device=device
    )

    # ---- Noise coefficient ----
    noisecoef_tensor = torch.tensor(
        classes_df["Perlin_noise"].values, dtype=torch.float64, device=device
    )

    # ---- Start radius ----
    startrad_tensor = torch.tensor(
        classes_df["Center_rad"].values, dtype=torch.int32, device=device
    )

    # ---- Line width ----
    linewidth_tensor = torch.tensor(
        classes_df["line_width"].values, dtype=torch.float64, device=device
    )

    # Save all parameter tensors into a list
    classes = [
        vertex_tensor,
        norbits_tensor,
        ovalx_tensor,
        ovaly_tensor,
        freq1_tensor,
        freq2_tensor,
        noisecoef_tensor,
        startrad_tensor,
        linewidth_tensor,
    ]

    # Return classes and maximum values for norbits and nvertex needed for batch vatom generation
    return classes, norbits_max, nvertex_max


# Class that generates classes and its parameters given configuration and stores info into respective GPU
def gen_classes(
    config: SectionProxy, nclasses: int, device: int
) -> Tuple[list[torch.Tensor], int, int]:

    # Get vars from config
    nvertex_min = config.getint("nvertex_min")
    nvertex_max = config.getint("nvertex_max")
    norbits_min = config.getint("norbits_min")
    norbits_max = config.getint("norbits_max")
    oval_max = config.getint("oval_max")
    freq_min = config.getint("freq_min")
    freq_max = config.getint("freq_max")
    noisecoef_min = config.getint("noisecoef_min")
    startrad_min = config.getint("startrad_min")
    linewidth_max = config.getfloat("linewidth_max")
    seed = config.getint("seed")

    # Torch generator (ensures exactly identical values across GPUs)
    g = torch.Generator(device=device).manual_seed(seed)

    # ---- Vertex ----
    vertex_tensor = torch.randint(
        low=nvertex_min,
        high=nvertex_max + 1,
        size=(nclasses,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    nvertex_max = torch.max(vertex_tensor).cpu().item()

    # ---- Orbits ----
    norbits_tensor = torch.randint(
        low=norbits_min,
        high=norbits_max + 1,
        size=(nclasses,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    norbits_max = torch.max(norbits_tensor).cpu().item()

    # ---- Oval rates ----
    ovalx_tensor = (
        torch.rand((nclasses,), generator=g, device=device, dtype=torch.float64)
        * (oval_max - 1)
        + 1
    )

    ovaly_tensor = (
        torch.rand((nclasses,), generator=g, device=device, dtype=torch.float64)
        * (oval_max - 1)
        + 1
    )

    # ---- Frequences: freq1 and freq2 (with freq2 != freq1 unless freq1=0) ----
    if freq_min == freq_max:
        freq1_tensor = torch.ones(nclasses, device=device, dtype=torch.int32)*freq_min
        freq2_tensor = torch.ones(nclasses, device=device, dtype=torch.int32)*freq_min
    else:

        freq1_tensor = torch.randint(
        low=freq_min,
        high=freq_max + 1,
        size=(nclasses,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )

        # freq2: generate candidates
        freq2_tensor = torch.randint(
            low=freq_min,
            high=freq_max + 1,
            size=(nclasses,),
            generator=g,
            device=device,
            dtype=torch.int32,
        )

        # freq2: enforce freq2 != freq1 unless freq1 == 0
        mask = (freq2_tensor == freq1_tensor) & (freq1_tensor != 0)
        while mask.any():
            freq2_tensor[mask] = torch.randint(
                low=freq_min,
                high=freq_max + 1,
                size=(mask.sum().item(),),
                generator=g,
                device=device,
                dtype=torch.int32,
            )
            mask = (freq2_tensor == freq1_tensor) & (freq1_tensor != 0)

    # ---- Noise coefficient ----
    noisecoef_tensor = (
        torch.rand((nclasses,), generator=g, device=device, dtype=torch.float64) * 4.0
        + noisecoef_min
    )

    # ---- Start radius ----
    startrad_tensor = torch.randint(
        low=startrad_min,
        high=startrad_min + 51,
        size=(nclasses,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )

    # ---- Line width ----
    linewidth_tensor = (
        torch.rand((nclasses,), generator=g, device=device, dtype=torch.float64)
        * linewidth_max
    )

    # Save all parameter tensors into a list
    classes = [
        vertex_tensor,
        norbits_tensor,
        ovalx_tensor,
        ovaly_tensor,
        freq1_tensor,
        freq2_tensor,
        noisecoef_tensor,
        startrad_tensor,
        linewidth_tensor,
    ]

    # Return classes and maximum values for norbits and nvertex needed for batch vatom generation
    return classes, norbits_max, nvertex_max
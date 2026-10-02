from typing import Tuple

import pandas as pd
from configparser import ConfigParser, SectionProxy
import torch
from torch.utils.data import IterableDataset, get_worker_info


## Basic dataset for vatoms
class VatomDataset(IterableDataset):
    def __init__(
        self,
        init_datapoints: int,
        batch_size: int,
        nclasses: int,
        device: int,  # GPU rank
        gpus: int,  # total GPUs
        aug_repeats: int = 1,  # GPUs sharing same sample stream
    ):
        self.init_datapoints = init_datapoints
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.device = device
        self.gpus = gpus
        self.aug_repeats = aug_repeats

        assert gpus % aug_repeats == 0, "aug_repeats must divide gpus"

    def generate_label(self, idx: int) -> torch.Tensor:
        labels = []
        for i in range(self.batch_size):
            g = torch.Generator().manual_seed(idx * self.batch_size + i)
            l = torch.randint(
                low=0,
                high=self.nclasses,
                size=(1,),
                generator=g,
            )
            labels.append(l)
        return torch.cat(labels, dim=0)

    def __iter__(self):

        # Logical stream ID: GPUs in the same repeat group share samples
        stream_id = self.device // self.aug_repeats
        num_streams = self.gpus // self.aug_repeats

        idx = self.init_datapoints + stream_id

        while True:
            yield idx, self.generate_label(idx)
            idx += num_streams


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


def read_classes(
    classes_df: pd.DataFrame, device: int
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


# Wrapper function to use from pretrain.py
def create_vatom_dataset(
    dataset_csv_path: str,
    dataset_cfg_path: str,
    dataset_cfg_select: str,
    init_datapoints: int,
    batch_size: int,
    nclasses: int,
    aug_repeats: int,
    device: int,
    gpus: int,
) -> IterableDataset:
    """Function that creates a dataset of visual atoms"""

    if dataset_csv_path:

        # Read csv with class parameters
        classes_df = pd.read_csv(dataset_csv_path)
        # Process data
        classes, norbits_max, nvertex_max = read_classes(classes_df, device=device)
    elif dataset_cfg_path:
        # Read config file
        configparser = ConfigParser()
        configparser.read(dataset_cfg_path)
        config = configparser[dataset_cfg_select]

        # Create classes
        classes, norbits_max, nvertex_max = gen_classes(
            config=config, nclasses=nclasses, device=device
        )
    else:
        raise (
            NotImplementedError,
            "Dataset csv with parameters or dataset cfg is required",
        )

    # Create dataset
    dataset = VatomDataset(
        init_datapoints=init_datapoints,
        batch_size=batch_size,
        nclasses=nclasses,
        aug_repeats=aug_repeats,
        device=device,
        gpus=gpus,
    )

    return dataset, classes, norbits_max, nvertex_max

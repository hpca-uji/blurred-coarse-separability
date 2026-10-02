# System imports
import argparse
from statistics import mean

# General external imports
import numpy as np
import pandas as pd
from configparser import ConfigParser

# Torch imports
import torch

# Local imports
from factory.extractor import build_easy_model
from factory.gpuload import read_classes, gen_classes, create_vatom_loader
from factory.metrics import compute_mi, compute_vendi


def load_args():
    
    # Create parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--kernel-file",
        type=str,
        required=True,
        help="Path to kernel code in GPU generation",
    )
    parser.add_argument(
        "--gpugen-config",
        type=str,
        required=True,
        help="Path to gpu generation configuration. Either .csv with the classes definition or .cfg with class generation parameters"
    )
    parser.add_argument(
        "--cfg-select",
        type=str,
        required=False,
        help="Selection of configuration inside .cfg file"
    )
    parser.add_argument(
        "--kernel-res",
        type=int,
        default=512,
        help="Resolution of generated images",
    )
    parser.add_argument(
        "--min-rand-kernel-res",
        type=int,
        default=0,
        help="Minimum kernel resolution when using random resolution",
    )
    parser.add_argument(
        "--nclasses",
        type=int,
        default=1000,
        help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=128,
        help="Number of images per class in the group",
    )

    # Common parameters
    parser.add_argument(
        "--model-res", 
        type=int, 
        default=224, 
        help="Input resolution of the pretrained model"
    )
    parser.add_argument(
        "--prep",
        type=str,
        choices=["crop", "resize", "none"],
        help="Whether to apply random crop, resize or none during data preprocessing",
    )

    # Parse args
    args = parser.parse_args()

    # Parameter validation
    if ".cfg" in args.gpugen_config and not args.cfg_select:
        raise ValueError("If .cfg file is passed as configuration, you need to select a specific configuration inside the file")
    elif ".csv" in args.gpugen_config and args.cfg_select:
        raise UserWarning("cfg select will be ignored since csv file was passed as configuration")

    return args


def build_classes(args, nclasses, device):

    if ".csv" in args.gpugen_config:
        classes_df = pd.read_csv(args.gpugen_config)
        classes, norbits_max, nvertex_max = read_classes(classes_df, device=device)
    elif ".cfg" in args.gpugen_config:
        configparser = ConfigParser()
        configparser.read(args.gpugen_config)
        config = configparser[args.cfg_select]
        classes, norbits_max, nvertex_max = gen_classes(
            config=config, nclasses=nclasses, device=device
        )
    else:
        raise NotImplementedError("Dataset csv with parameters or dataset cfg is required")

    return classes, norbits_max, nvertex_max


def main():

    # Load args
    args = load_args()

    # Get device
    device = torch.device(0)

    # Build model
    model = build_easy_model(device=device)

    # Build classes
    classes, norbits_max, nvertex_max = build_classes(args, args.nclasses, device)
    
    # Create dataset
    loader = create_vatom_loader(
        kernel_file=args.kernel_file,
        device=device,
        samples_per_class=args.samples_per_class,
        nclasses=args.nclasses,
        classes=classes,
        nvertex_max=nvertex_max,
        norbits_max=norbits_max,
        res=args.model_res,
        prep=args.prep,
        kernel_res=args.kernel_res,
        min_rand_kernel_res=args.min_rand_kernel_res
    )
    
    # Extract features
    features = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i%10 == 0:
                print(f"{i}/{len(loader)}", flush=True)
            feats = model(batch)
            features.append(feats.cpu())

    # Clear
    loader.clear_ctx()
    del loader

    # Create numpy array [num_classes, samples_per_class, embedding_dim]
    features = np.array(features, dtype=object)

    # Compute mi
    compute_mi(features=features)

    # Compute vendi
    compute_vendi(features=features)


if __name__ == "__main__":
    main()
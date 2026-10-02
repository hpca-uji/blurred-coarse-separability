# System imports
import argparse
from statistics import mean

# torch imports
import torch

# torchvision imports
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    Normalize,
    ToDtype,
)

# timm imports
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Local imports
from factory.fileload import compute_file_features
from factory.extractor import build_easy_model
from factory.metrics import compute_mi, compute_vendi


def load_args():
    
    # Create parser
    parser = argparse.ArgumentParser()

    # Add arguments
    
    # Arguments for class-based folder structure files
    parser.add_argument(
        "datadir", 
        type=str, 
        help="Path to class-based folder structure")
    parser.add_argument(
        "--total-files",
        type=int,
        default=128000,
        help="Total number of images to process across all classes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible image selection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
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

    # Validate parameters
    if args.total_files <= 0:
        raise ValueError("--total-files must be greater than 0.")

    return args


def build_transform(model_res, prep):

    # Select crop/resize/none
    if prep == "crop":
        resize = RandomResizedCropAndInterpolation(size=model_res)
    elif prep == "resize":
        resize = Resize(size=(model_res, model_res))
    elif prep == "none":
        pass
    else:
        raise ValueError("Select 'crop', 'resize' or 'none'")

    # Create transforms list
    transforms = [
        resize,
        ToDtype(torch.float32, scale=True),
    ]

    # Add normalization
    transforms.append(
        Normalize(
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD,
        )
    )

    return Compose(transforms)

def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def main():

    # Load args
    args = load_args()

    # Get device
    device = get_device()

    # Build model
    model = build_easy_model(device=device)

    # Build transform
    transform = build_transform(model_res=args.model_res, 
                                prep=args.prep)

    # Compute features
    features = compute_file_features(model=model,
                                     transform=transform,
                                     datadir=args.datadir, 
                                     total_files=args.total_files, 
                                     seed=args.seed,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     device=device)

    # Compute mi
    compute_mi(features=features)

    # Compute vendi
    compute_vendi(features=features)

if __name__ == "__main__":
    main()
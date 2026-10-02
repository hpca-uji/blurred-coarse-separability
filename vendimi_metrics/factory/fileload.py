# Disclaimer: Partially put together by AI

# general imports
import os

# Other troncal imports
import numpy as np

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# timm imports
from torchvision.io import read_image


# -----------------------------
# Dataset
# -----------------------------
class ImageSubsetDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        img = read_image(path)

        if self.transform:
            img = self.transform(img)

        return img


# -----------------------------
# Feature extraction
# -----------------------------
def extract_class_features(
    model,
    image_paths,
    transform,
    batch_size,
    num_workers,
    device,
):
    """
    Extract features for a single class.

    Returns:
        np.ndarray with shape:
        (num_images, feature_dim)
    """
    dataset = ImageSubsetDataset(
        image_paths,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    features = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(
                device,
                non_blocking=True,
            )

            feats = model(batch)

            features.append(
                feats.cpu().numpy()
            )

    if not features:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    return np.concatenate(
        features,
        axis=0,
    )


# -----------------------------
# Random sampling
# -----------------------------
def randomly_select_paths(
    class_paths,
    class_names,
    allocations,
    seed,
):
    """
    Randomly select the requested number of images
    from each class without replacement.

    Sampling is reproducible for a given seed.
    """
    rng = np.random.default_rng(seed)

    selected_paths = []

    for class_name, n_samples in zip(
        class_names,
        allocations,
    ):
        paths = class_paths[class_name]

        if n_samples == 0:
            selected_paths.append([])
            continue

        indices = rng.choice(
            len(paths),
            size=n_samples,
            replace=False,
        )

        selected = [
            paths[i]
            for i in indices
        ]

        selected_paths.append(selected)

    return selected_paths


# -----------------------------
# Allocate samples
# -----------------------------
def allocate_samples(class_counts, total_files):
    """
    Allocate total_files across classes according to
    the original dataset proportions.

    Uses the largest-remainder method so that the
    allocations sum exactly to total_files.
    """
    class_counts = np.asarray(
        class_counts,
        dtype=np.int64,
    )

    total_available = class_counts.sum()

    proportions = class_counts / total_available
    exact = proportions * total_files

    # Start with the floor of each allocation.
    allocations = np.floor(exact).astype(np.int64)

    # Distribute the remaining samples according to
    # the largest fractional remainders.
    remainder = total_files - allocations.sum()

    if remainder > 0:
        fractional = exact - allocations
        indices = np.argsort(-fractional)

        allocations[indices[:remainder]] += 1

    # Sanity checks.
    assert allocations.sum() == total_files
    assert np.all(allocations >= 0)
    assert np.all(allocations <= class_counts)

    return allocations


# -----------------------------
# Collect image paths
# -----------------------------
def get_class_paths(datadir, class_names):
    class_paths = {}

    for class_name in class_names:
        class_dir = os.path.join(datadir, class_name)

        files = sorted(
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        )

        class_paths[class_name] = [
            os.path.join(class_dir, f)
            for f in files
        ]

    return class_paths


# -----------------------------
# Get class names
# -----------------------------
def get_classnames(datadir):

    class_names = sorted(
        [
            c
            for c in os.listdir(datadir)
            if os.path.isdir(
                os.path.join(
                    datadir,
                    c,
                )
            )
        ]
    )
    return class_names


def compute_file_features(model, transform, datadir, total_files, seed, batch_size, num_workers, device):

    # Get classnames inside datadir
    class_names = get_classnames(datadir)

    # Check that there are classes folders inside datadir
    if not class_names:
        raise ValueError(f"No class directories found in f{datadir}")

    # Get class_paths
    class_paths = get_class_paths(datadir, class_names)

    # Compute number of files per class
    class_counts = [
        len(class_paths[class_name])
        for class_name in class_names
    ]

    # Compute total number of files and check requested files
    total_available = sum(class_counts)
    if total_available < total_files:
        raise ValueError(f"Requested {total_files} files, but only {total_available} files are available.")

    # Print stats
    print(f"Found {len(class_names)} classes")
    print(f"Total available files: {total_available}")
    print(f"Requested files: {total_files}")
    print(f"Random seed: {seed}")

    # Allocate proportionally
    allocations = allocate_samples(class_counts, total_files)

    # Print stats by class
    print("\nSample allocation:")
    for class_name, available, n_samples in zip(
        class_names,
        class_counts,
        allocations,
    ):
        proportion = available / total_available
        print(f"{class_name}: {available} available ({proportion:.2%}) -> {n_samples} selected")

    # Randomly select images
    selected_paths = randomly_select_paths(
        class_paths=class_paths,
        class_names=class_names,
        allocations=allocations,
        seed=seed,
    )

    # Extract features per class
    features_list = []

    for i, (class_name, paths) in enumerate(zip(class_names,selected_paths)):

        # Print progress
        print(f"\n{i/len(class_names)*100:.0f}% Extracting {len(paths)} images from '{class_name}'")

        # Extract class features
        class_features = extract_class_features(
            model=model,
            image_paths=paths,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        # Add to list
        features_list.append(
            class_features
        )

    # Build empty object array
    features = np.empty(
        len(features_list),
        dtype=object,
        )

    # Fill object array
    for i, class_features in enumerate(features_list):
        features[i] = class_features


    return features
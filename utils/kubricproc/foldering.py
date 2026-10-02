from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Where your downloaded TFDS dataset is located.
#
# Example:
# SOURCE_DIR = "/home/myuser/datasets/kubric"
#
# The directory should contain:
#
#   shapenet_pretraining/
#       dataset_info.json
#       features.json
#       label.labels.txt
#       shapenet_pretraining-train-...
#       shapenet_pretraining-validation-...
# ------------------------------------------------------------

SOURCE_DIR = "."


# ------------------------------------------------------------
# Where you want the new simple dataset to be created.
#
# Example:
# OUTPUT_DIR = "/home/myuser/datasets/shapenet_simple"
# ------------------------------------------------------------

OUTPUT_DIR = "shapenet_pretraining_folders"


# ============================================================
# SHAPENET CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "airplane",
    "ashcan",
    "bag",
    "basket",
    "bathtub",
    "bed",
    "bench",
    "birdhouse",
    "bookshelf",
    "bottle",
    "bowl",
    "bus",
    "cabinet",
    "camera",
    "can",
    "cap",
    "car",
    "cellular_telephone",
    "chair",
    "clock",
    "computer_keyboard",
    "dishwasher",
    "display",
    "earphone",
    "faucet",
    "file",
    "guitar",
    "helmet",
    "jar",
    "knife",
    "lamp",
    "laptop",
    "loudspeaker",
    "mailbox",
    "microphone",
    "microwave",
    "motorcycle",
    "mug",
    "piano",
    "pillow",
    "pistol",
    "pot",
    "printer",
    "remote_control",
    "rifle",
    "rocket",
    "skateboard",
    "sofa",
    "stove",
    "table",
    "telephone",
    "tower",
    "train",
    "vessel",
    "washer",
]


# ============================================================
# CHECK CONFIGURATION
# ============================================================

SOURCE_DIR = Path(SOURCE_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source directory does not exist:\n{SOURCE_DIR}"
    )

if len(CLASS_NAMES) != 55:
    raise ValueError(
        f"Expected 55 classes, but found {len(CLASS_NAMES)}"
    )


print("=" * 70)
print("ShapeNet TFDS -> Simple Image Dataset")
print("=" * 70)

print(f"Source directory:")
print(f"  {SOURCE_DIR}")

print(f"\nOutput directory:")
print(f"  {OUTPUT_DIR}")

print(f"\nNumber of classes:")
print(f"  {len(CLASS_NAMES)}")

print("=" * 70)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val"]:

    for class_name in CLASS_NAMES:

        class_dir = OUTPUT_DIR / split / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


print("\nOutput directory structure created.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_ds = tfds.load(
    "shapenet_pretraining",
    data_dir=str(SOURCE_DIR),
    split="train",
    shuffle_files=False,
)

print("Training dataset loaded.")

print("\nLoading validation dataset...")

val_ds = tfds.load(
    "shapenet_pretraining",
    data_dir=str(SOURCE_DIR),
    split="validation",
    shuffle_files=False,
)

print("Validation dataset loaded.")


# ============================================================
# CONVERSION FUNCTION
# ============================================================

def convert_dataset(dataset, split_name):
    """
    Convert one TFDS split into:

        output/
            split_name/
                class_name/
                    image_id.png
    """

    print("\n")
    print("=" * 70)
    print(f"Converting {split_name} dataset")
    print("=" * 70)

    # Number of images written for each class
    counters = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    # Number of images skipped because they already exist
    skipped = 0

    # Total examples processed
    total = 0

    for example in dataset:

        total += 1

        # ----------------------------------------------------
        # Get label
        # ----------------------------------------------------

        label = int(
            example["label"].numpy()
        )

        if label < 0 or label >= len(CLASS_NAMES):

            raise ValueError(
                f"Invalid label {label}. "
                f"Expected 0-{len(CLASS_NAMES)-1}."
            )

        class_name = CLASS_NAMES[label]


        # ----------------------------------------------------
        # Get image ID
        # ----------------------------------------------------

        image_id = (
            example["image_id"]
            .numpy()
            .decode("utf-8")
        )


        # ----------------------------------------------------
        # Get image
        # ----------------------------------------------------

        image_array = example["image"].numpy()

        # Expected shape:
        #
        #   (height, width, 4)
        #
        # because the dataset contains RGBA images.

        if image_array.ndim != 3:
            raise ValueError(
                f"Unexpected image shape: "
                f"{image_array.shape}"
            )

        if image_array.shape[-1] != 4:
            raise ValueError(
                f"Expected RGBA image with 4 channels, "
                f"but got shape {image_array.shape}"
            )


        # ----------------------------------------------------
        # Create output filename
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / split_name
            / class_name
            / f"{image_id}.png"
        )


        # ----------------------------------------------------
        # Skip existing images
        #
        # This makes it possible to restart the script
        # after an interruption without rewriting everything.
        # ----------------------------------------------------

        if output_path.exists():

            skipped += 1

        else:

            # ------------------------------------------------
            # Convert NumPy array -> PIL RGBA image
            # ------------------------------------------------

            image = Image.fromarray(
                image_array,
                mode="RGBA",
            )

            # ------------------------------------------------
            # Save as PNG
            # ------------------------------------------------

            image.save(
                output_path,
                format="PNG",
            )

            counters[class_name] += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if total % 1000 == 0:

            print(
                f"{split_name}: "
                f"{total:,} examples processed | "
                f"{total - skipped:,} saved | "
                f"{skipped:,} skipped"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(f"{split_name.upper()} COMPLETE")
    print("=" * 70)

    print(f"Total examples processed: {total:,}")
    print(f"New images written:       {sum(counters.values()):,}")
    print(f"Existing images skipped:  {skipped:,}")

    print("\nImages written per class:")

    for label, class_name in enumerate(CLASS_NAMES):

        count = counters[class_name]

        print(
            f"{label:2d}  "
            f"{class_name:25s} "
            f"{count:,}"
        )


# ============================================================
# CONVERT TRAINING DATA
# ============================================================

convert_dataset(
    train_ds,
    "train",
)


# ============================================================
# CONVERT VALIDATION DATA
# ============================================================

convert_dataset(
    val_ds,
    "val",
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nDone!")

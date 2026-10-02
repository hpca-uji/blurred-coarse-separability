# Disclaimer: Mainly build using generative AI

from pathlib import Path
import pandas as pd
import io
from PIL import Image


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_DIR = Path("input/path")
OUTPUT_DIR = Path("output/path")

# Set to True if you want to verify that the bytes are actually valid
# images. False is considerably faster.
VERIFY_IMAGES = False


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_index = 0

parquet_files = sorted(INPUT_DIR.glob("*.parquet"))

print(f"Found {len(parquet_files)} parquet files")

for parquet_idx, parquet_file in enumerate(parquet_files):

    print(
        f"\n[{parquet_idx + 1}/{len(parquet_files)}] "
        f"Processing {parquet_file}"
    )

    df = pd.read_parquet(parquet_file)

    print(f"  Number of images: {len(df)}")

    for row_idx, (image_data, label) in enumerate(
        zip(df["image"], df["label"])
    ):

        # -------------------------------------------------------------
        # Get class name
        # -------------------------------------------------------------

        class_name = str(label)

        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Extract image bytes
        # -------------------------------------------------------------

        image_bytes = image_data["bytes"]

        if image_bytes is None:
            print(
                f"  WARNING: image {row_idx} has no bytes, skipping"
            )
            continue

        # -------------------------------------------------------------
        # Determine image format
        # -------------------------------------------------------------

        if VERIFY_IMAGES:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.verify()

                extension = (
                    image.format.lower()
                    if image.format is not None
                    else "jpg"
                )

            except Exception as e:
                print(
                    f"  WARNING: invalid image at row {row_idx}: {e}"
                )
                continue

        else:
            # Your example starts with FF D8 FF, i.e. JPEG.
            # We can determine the extension from the magic bytes.
            if image_bytes.startswith(b"\xff\xd8\xff"):
                extension = "jpg"
            elif image_bytes.startswith(b"\x89PNG"):
                extension = "png"
            elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
                extension = "webp"
            elif image_bytes.startswith(b"GIF8"):
                extension = "gif"
            else:
                extension = "bin"

        # -------------------------------------------------------------
        # Create unique filename
        # -------------------------------------------------------------

        output_file = (
            class_dir / f"image_{global_index:09d}.{extension}"
        )

        # -------------------------------------------------------------
        # Write bytes directly
        # -------------------------------------------------------------

        with open(output_file, "wb") as f:
            f.write(image_bytes)

        global_index += 1

    # Free the DataFrame before moving to the next shard
    del df

    print(f"  Total extracted so far: {global_index}")


print("\nDone!")
print(f"Total images extracted: {global_index}")
print(f"Output directory: {OUTPUT_DIR}")
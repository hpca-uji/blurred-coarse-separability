#!/bin/bash

set -euo pipefail

###############################################################################
# downsize_dataset.sh
# DISCLAIMER: AI-Generated
# Downsize a class-organized image dataset while preserving class proportions,
# or reproduce an existing reduced dataset from a manifest.
#
# Assumption:
#   Each immediate subdirectory of --input is a class.
#
# Example:
#
#   imagenet/train/
#       n01440764/
#       n01443537/
#       n01484850/
#       ...
#
# Two operating modes:
#
#   1. Sampling mode
#
#      --num-images N
#
#      Select N images proportionally across classes and generate a manifest.
#
#   2. Manifest mode
#
#      --manifest FILE
#
#      Recreate exactly the files listed in the manifest.
#
# Manifest paths are ALWAYS relative to --input.
#
###############################################################################

INPUT=""
OUTPUT=""
TARGET=""
MANIFEST_INPUT=""
SEED=12345
WORKERS=1
MODE="symlink"

###############################################################################
# Usage
###############################################################################

usage() {
    cat <<EOF

Usage:

SAMPLING MODE
=============

  $0 \\
      --input DIR \\
      --output DIR \\
      --num-images N \\
      [OPTIONS]


MANIFEST MODE
=============

  $0 \\
      --input DIR \\
      --output DIR \\
      --manifest FILE \\
      [OPTIONS]


Required:

  --input DIR
        Original dataset directory.

  --output DIR
        Destination directory.

  --num-images N
        Number of images to select.
        Mutually exclusive with --manifest.

  OR

  --manifest FILE
        Existing manifest to reproduce.
        Mutually exclusive with --num-images.


Optional:

  --seed N
        Random seed used in sampling mode.
        Default: 12345

  --workers N
        Number of parallel workers.
        Default: 1

  --mode MODE
        symlink
            Create symbolic links to original images.

        copy
            Physically copy images.

        manifest
            Generate/use the manifest without creating image files.

        Default: symlink

  --help
        Show this help.


EXAMPLES
========

Reduce ImageNet to exactly one million images:

  $0 \\
      --input /data/imagenet/train \\
      --output /data/imagenet_1m \\
      --num-images 1000000 \\
      --seed 12345 \\
      --workers 32 \\
      --mode symlink


Reduce Places365 to one million images by copying:

  $0 \\
      --input /data/places365/train \\
      --output /data/places365_1m \\
      --num-images 1000000 \\
      --seed 12345 \\
      --workers 32 \\
      --mode copy


Reproduce an existing subset from its manifest:

  $0 \\
      --input /data/imagenet/train \\
      --output /scratch/imagenet_1m \\
      --manifest /data/imagenet_1m/selected_files.tsv \\
      --mode symlink \\
      --workers 32


Reproduce it physically:

  $0 \\
      --input /data/imagenet/train \\
      --output /scratch/imagenet_1m \\
      --manifest /data/imagenet_1m/selected_files.tsv \\
      --mode copy \\
      --workers 32


Check/create only the manifest:

  $0 \\
      --input /data/imagenet/train \\
      --output /data/imagenet_1m \\
      --manifest /data/imagenet_1m/selected_files.tsv \\
      --mode manifest

EOF
}

###############################################################################
# Error helper
###############################################################################

die() {
    echo "ERROR: $*" >&2
    exit 1
}

###############################################################################
# Argument parsing
###############################################################################

while [[ $# -gt 0 ]]; do

    case "$1" in

        --input)
            [[ $# -ge 2 ]] || die "--input requires an argument"
            INPUT="$2"
            shift 2
            ;;

        --output)
            [[ $# -ge 2 ]] || die "--output requires an argument"
            OUTPUT="$2"
            shift 2
            ;;

        --num-images)
            [[ $# -ge 2 ]] || die "--num-images requires an argument"
            TARGET="$2"
            shift 2
            ;;

        --manifest)
            [[ $# -ge 2 ]] || die "--manifest requires an argument"
            MANIFEST_INPUT="$2"
            shift 2
            ;;

        --seed)
            [[ $# -ge 2 ]] || die "--seed requires an argument"
            SEED="$2"
            shift 2
            ;;

        --workers)
            [[ $# -ge 2 ]] || die "--workers requires an argument"
            WORKERS="$2"
            shift 2
            ;;

        --mode)
            [[ $# -ge 2 ]] || die "--mode requires an argument"
            MODE="$2"
            shift 2
            ;;

        --help|-h)
            usage
            exit 0
            ;;

        *)
            die "Unknown argument: $1"
            ;;

    esac

done

###############################################################################
# Validate arguments
###############################################################################

[[ -n "$INPUT" ]] || die "--input is required"
[[ -n "$OUTPUT" ]] || die "--output is required"

[[ -d "$INPUT" ]] || die "Input directory does not exist: $INPUT"

if [[ -n "$TARGET" && -n "$MANIFEST_INPUT" ]]; then
    die "--num-images and --manifest are mutually exclusive"
fi

if [[ -z "$TARGET" && -z "$MANIFEST_INPUT" ]]; then
    die "Specify either --num-images or --manifest"
fi

if [[ -n "$TARGET" ]]; then
    [[ "$TARGET" =~ ^[0-9]+$ ]] \
        || die "--num-images must be a non-negative integer"
fi

[[ "$WORKERS" =~ ^[0-9]+$ ]] \
    || die "--workers must be a non-negative integer"

[[ "$SEED" =~ ^[0-9]+$ ]] \
    || die "--seed must be a non-negative integer"

case "$MODE" in
    symlink|copy|manifest)
        ;;
    *)
        die "--mode must be symlink, copy, or manifest"
        ;;
esac

[[ "$WORKERS" -ge 1 ]] \
    || die "--workers must be at least 1"

###############################################################################
# Required programs
###############################################################################

command -v python3 >/dev/null 2>&1 \
    || die "python3 not found"

command -v find >/dev/null 2>&1 \
    || die "find not found"

###############################################################################
# GNU parallel is optional.
###############################################################################

HAVE_PARALLEL=0

if command -v parallel >/dev/null 2>&1; then
    HAVE_PARALLEL=1
fi

if [[ "$HAVE_PARALLEL" -eq 0 && "$WORKERS" -gt 1 ]]; then
    echo
    echo "WARNING: GNU parallel was not found."
    echo "         Falling back to one worker."
    echo
    WORKERS=1
fi

###############################################################################
# Create output directory
###############################################################################

mkdir -p "$OUTPUT"

###############################################################################
# Temporary directory
###############################################################################

TMPDIR=$(mktemp -d)

cleanup() {
    rm -rf "$TMPDIR"
}

trap cleanup EXIT

###############################################################################
# Output files
###############################################################################

OUTPUT_MANIFEST="$OUTPUT/selected_files.tsv"
CLASS_COUNTS="$OUTPUT/class_counts.tsv"
CLASS_ALLOCATIONS="$OUTPUT/class_allocations.tsv"
DATASET_INFO="$OUTPUT/dataset_info.txt"

###############################################################################
# Image extensions
###############################################################################

is_image_find_expression() {
    echo \
        "\( " \
        "-iname '*.jpg' -o " \
        "-iname '*.jpeg' -o " \
        "-iname '*.png' -o " \
        "-iname '*.webp' -o " \
        "-iname '*.bmp' -o " \
        "-iname '*.tif' -o " \
        "-iname '*.tiff' " \
        "\)"
}

###############################################################################
# Resolve input directory to an absolute path.
#
# This is important because manifests contain relative paths only.
###############################################################################

INPUT=$(cd "$INPUT" && pwd)

###############################################################################
# Banner
###############################################################################

echo
echo "============================================================"
echo "Dataset downsizing / reproduction"
echo "============================================================"
echo "Input:       $INPUT"
echo "Output:      $OUTPUT"
echo "Workers:     $WORKERS"
echo "Mode:        $MODE"

if [[ -n "$TARGET" ]]; then
    echo "Operation:   proportional sampling"
    echo "Target:      $TARGET images"
    echo "Seed:        $SEED"
else
    echo "Operation:   manifest reproduction"
    echo "Manifest:    $MANIFEST_INPUT"
fi

echo "============================================================"
echo

###############################################################################
# ============================================================================
# MODE 1: MANIFEST REPRODUCTION
# ============================================================================
###############################################################################

if [[ -n "$MANIFEST_INPUT" ]]; then

    [[ -f "$MANIFEST_INPUT" ]] \
        || die "Manifest does not exist: $MANIFEST_INPUT"

    echo "Manifest reproduction mode"
    echo

    ###########################################################################
    # Validate and normalize manifest
    #
    # Expected format:
    #
    # class<TAB>relative/path/to/image.jpg
    #
    ###########################################################################

    NORMALIZED_MANIFEST="$TMPDIR/normalized_manifest.tsv"

    python3 - "$MANIFEST_INPUT" "$INPUT" "$NORMALIZED_MANIFEST" <<'PY'
import sys
import os

manifest = sys.argv[1]
input_root = sys.argv[2]
output = sys.argv[3]

entries = []
errors = []

with open(
    manifest,
    "r",
    encoding="utf-8",
    errors="surrogateescape"
) as f:

    for line_number, line in enumerate(f, 1):

        line = line.rstrip("\n")

        if not line:
            continue

        parts = line.split("\t")

        if len(parts) != 2:
            errors.append(
                f"line {line_number}: expected 2 tab-separated fields"
            )
            continue

        class_name, relative_path = parts

        # Manifest paths must be relative.
        if os.path.isabs(relative_path):
            errors.append(
                f"line {line_number}: path is absolute: "
                f"{relative_path}"
            )
            continue

        # Normalize the path.
        normalized = os.path.normpath(relative_path)

        # Reject paths that escape the dataset root.
        if normalized == ".." or normalized.startswith("../"):
            errors.append(
                f"line {line_number}: path escapes input directory: "
                f"{relative_path}"
            )
            continue

        full_path = os.path.join(input_root, normalized)

        if not os.path.isfile(full_path):
            errors.append(
                f"line {line_number}: file does not exist: "
                f"{relative_path}"
            )
            continue

        entries.append((class_name, normalized))

if errors:
    print(
        f"Manifest validation failed with {len(errors)} error(s):",
        file=sys.stderr
    )

    for error in errors[:100]:
        print("  " + error, file=sys.stderr)

    if len(errors) > 100:
        print(
            f"  ... and {len(errors) - 100} more",
            file=sys.stderr
        )

    sys.exit(1)

# Sort for deterministic output.
entries.sort(key=lambda x: (x[0], x[1]))

# Remove exact duplicate entries.
unique_entries = []
previous = None

for entry in entries:
    if entry != previous:
        unique_entries.append(entry)
    previous = entry

with open(
    output,
    "w",
    encoding="utf-8",
    errors="surrogateescape"
) as f:

    for class_name, relative_path in unique_entries:
        f.write(f"{class_name}\t{relative_path}\n")

print(f"Validated {len(unique_entries)} manifest entries.")
PY

    cp "$NORMALIZED_MANIFEST" "$OUTPUT_MANIFEST"

    MANIFEST_COUNT=$(wc -l < "$OUTPUT_MANIFEST")

    echo
    echo "Manifest entries: $MANIFEST_COUNT"
    echo

    ###########################################################################
    # Reproduce files
    ###########################################################################

    if [[ "$MODE" != "manifest" ]]; then

        echo "Creating output dataset..."

        while IFS=$'\t' read -r class_name relative_path; do

            source="$INPUT/$relative_path"
            destination="$OUTPUT/$relative_path"

            mkdir -p "$(dirname "$destination")"

            if [[ "$MODE" == "symlink" ]]; then

                ln -s "$source" "$destination"

            elif [[ "$MODE" == "copy" ]]; then

                cp -- "$source" "$destination"

            fi

        done < "$OUTPUT_MANIFEST"

    fi

    ###########################################################################
    # Generate class allocation information from manifest
    ###########################################################################

    python3 - "$OUTPUT_MANIFEST" "$CLASS_ALLOCATIONS" <<'PY'
import sys
from collections import Counter

manifest = sys.argv[1]
output = sys.argv[2]

counts = Counter()

with open(
    manifest,
    "r",
    encoding="utf-8",
    errors="surrogateescape"
) as f:

    for line in f:
        line = line.rstrip("\n")

        if not line:
            continue

        class_name, relative_path = line.split("\t", 1)

        counts[class_name] += 1

with open(output, "w", encoding="utf-8") as f:

    for class_name in sorted(counts):
        f.write(f"{class_name}\t{counts[class_name]}\n")
PY

    ###########################################################################
    # Save metadata
    ###########################################################################

    cat > "$DATASET_INFO" <<EOF
Operation:
Manifest reproduction

Original dataset:
$INPUT

Output dataset:
$OUTPUT

Input manifest:
$MANIFEST_INPUT

Output manifest:
$OUTPUT_MANIFEST

Selected images:
$MANIFEST_COUNT

Mode:
$MODE

Manifest paths:
Relative to the input dataset root.

Random sampling:
Not performed.

The manifest is the authoritative definition of this dataset.
EOF

    echo
    echo "============================================================"
    echo "Manifest reproduction finished"
    echo "============================================================"
    echo "Images:       $MANIFEST_COUNT"
    echo "Manifest:     $OUTPUT_MANIFEST"
    echo "Allocations:   $CLASS_ALLOCATIONS"
    echo "Metadata:      $DATASET_INFO"
    echo "============================================================"
    echo

    exit 0
fi

###############################################################################
# ============================================================================
# MODE 2: PROPORTIONAL SAMPLING
# ============================================================================
###############################################################################

echo "Sampling mode"
echo

###############################################################################
# Temporary files for sampling
###############################################################################

TMP_CLASS_COUNTS="$TMPDIR/class_counts.tsv"
TMP_ALLOCATIONS="$TMPDIR/allocations.tsv"
TMP_CLASS_JOBS="$TMPDIR/class_jobs.tsv"

###############################################################################
# Step 1: Discover classes
###############################################################################

echo "Scanning dataset for classes..."

find "$INPUT" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -print0 |
while IFS= read -r -d '' class_dir; do

    class=$(basename "$class_dir")

    count=$(
        find "$class_dir" \
            -type f \
            \( \
                -iname '*.jpg' -o \
                -iname '*.jpeg' -o \
                -iname '*.png' -o \
                -iname '*.webp' -o \
                -iname '*.bmp' -o \
                -iname '*.tif' -o \
                -iname '*.tiff' \
            \) \
            -print |
        wc -l
    )

    printf '%s\t%s\n' "$class" "$count"

done > "$TMP_CLASS_COUNTS"

###############################################################################
# Basic statistics
###############################################################################

NUM_CLASSES=$(wc -l < "$TMP_CLASS_COUNTS")

TOTAL_IMAGES=$(
    awk -F '\t' '
        {
            sum += $2
        }
        END {
            print sum + 0
        }
    ' "$TMP_CLASS_COUNTS"
)

echo
echo "Classes found: $NUM_CLASSES"
echo "Images found:  $TOTAL_IMAGES"
echo

[[ "$NUM_CLASSES" -gt 0 ]] \
    || die "No class directories found under $INPUT"

[[ "$TARGET" -le "$TOTAL_IMAGES" ]] \
    || die \
        "Requested $TARGET images, but dataset contains only $TOTAL_IMAGES"

###############################################################################
# Step 2: Proportional allocation
###############################################################################

echo "Calculating proportional class allocations..."

python3 - \
    "$TMP_CLASS_COUNTS" \
    "$TMP_ALLOCATIONS" \
    "$TARGET" <<'PY'

import sys

counts_file = sys.argv[1]
output_file = sys.argv[2]
target = int(sys.argv[3])

classes = []

with open(
    counts_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.rstrip("\n")

        if not line:
            continue

        class_name, count = line.split("\t")

        classes.append(
            (class_name, int(count))
        )

total = sum(
    count for _, count in classes
)

if target > total:
    raise RuntimeError(
        f"Requested {target} images but only {total} exist"
    )

###############################################################################
# Largest-remainder method
###############################################################################

allocations = []

for class_name, count in classes:

    ideal = target * count / total

    base = int(ideal)

    remainder = ideal - base

    # Never request more images than the class contains.
    base = min(base, count)

    allocations.append(
        [
            class_name,
            count,
            ideal,
            base,
            remainder
        ]
    )

allocated = sum(
    x[3] for x in allocations
)

remaining = target - allocated

###############################################################################
# Classes with the largest fractional remainder get the leftover images.
###############################################################################

allocations.sort(
    key=lambda x: (-x[4], x[0])
)

for x in allocations:

    if remaining <= 0:
        break

    class_name, count, ideal, base, remainder = x

    if base < count:
        x[3] += 1
        remaining -= 1

if remaining != 0:
    raise RuntimeError(
        f"Could not allocate exactly {target} images. "
        f"{remaining} images remain."
    )

###############################################################################
# Return to alphabetical class order.
###############################################################################

allocations.sort(
    key=lambda x: x[0]
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for (
        class_name,
        count,
        ideal,
        allocation,
        remainder
    ) in allocations:

        f.write(
            f"{class_name}\t{count}\t{allocation}\n"
        )
PY

###############################################################################
# Step 3: Display allocation
###############################################################################

echo
echo "Class allocation:"
echo

printf "%-40s %12s %12s %12s\n" \
    "CLASS" \
    "ORIGINAL" \
    "SELECTED" \
    "PERCENT"

echo "--------------------------------------------------------------------------------"

awk -F '\t' -v total="$TOTAL_IMAGES" '
{
    pct = ($2 / total) * 100

    printf "%-40s %12d %12d %11.4f%%\n",
           $1,
           $2,
           $3,
           pct
}
' "$TMP_ALLOCATIONS"

SELECTED_TOTAL=$(
    awk -F '\t' '
        {
            sum += $3
        }
        END {
            print sum + 0
        }
    ' "$TMP_ALLOCATIONS"
)

echo "--------------------------------------------------------------------------------"

printf "%-40s %12d %12d\n" \
    "TOTAL" \
    "$TOTAL_IMAGES" \
    "$SELECTED_TOTAL"

echo

[[ "$SELECTED_TOTAL" -eq "$TARGET" ]] \
    || die \
        "Internal error: allocation contains $SELECTED_TOTAL images, expected $TARGET"

###############################################################################
# Save class statistics
###############################################################################

cp "$TMP_CLASS_COUNTS" "$CLASS_COUNTS"
cp "$TMP_ALLOCATIONS" "$CLASS_ALLOCATIONS"

###############################################################################
# Step 4: Create class jobs
###############################################################################

while IFS=$'\t' read -r class count selected; do

    class_dir="$INPUT/$class"

    printf '%s\t%s\t%s\t%s\n' \
        "$class" \
        "$count" \
        "$selected" \
        "$class_dir"

done < "$TMP_ALLOCATIONS" > "$TMP_CLASS_JOBS"

###############################################################################
# Step 5: Function to sample one class
###############################################################################

select_class() {

    local class="$1"
    local original_count="$2"
    local selected_count="$3"
    local class_dir="$4"

    local safe_class
    safe_class=$(printf '%s' "$class" | tr '/ ' '__')

    local class_tmp="$TMPDIR/files_${safe_class}.txt"
    local class_manifest="$TMPDIR/manifest_${safe_class}.tsv"

    ###########################################################################
    # Find all images in this class.
    #
    # IMPORTANT:
    #
    # We explicitly prepend the CLASS name.
    #
    # find -printf '%P' is relative to class_dir, so without the prefix:
    #
    #     image.jpg
    #
    # would be produced instead of:
    #
    #     class_name/image.jpg
    #
    ###########################################################################

    find "$class_dir" \
        -type f \
        \( \
            -iname '*.jpg' -o \
            -iname '*.jpeg' -o \
            -iname '*.png' -o \
            -iname '*.webp' -o \
            -iname '*.bmp' -o \
            -iname '*.tif' -o \
            -iname '*.tiff' \
        \) \
        -printf '%P\n' |
    sort |
    while IFS= read -r filename; do
        printf '%s/%s\n' "$class" "$filename"
    done > "$class_tmp"

    ###########################################################################
    # Sanity check
    ###########################################################################

    discovered=$(wc -l < "$class_tmp")

    if [[ "$discovered" -ne "$original_count" ]]; then

        echo
        echo "ERROR: class count changed while processing."
        echo "Class:       $class"
        echo "Expected:    $original_count"
        echo "Discovered:  $discovered"
        echo

        return 1
    fi

    ###########################################################################
    # Empty selection
    ###########################################################################

    if [[ "$selected_count" -eq 0 ]]; then
        : > "$class_manifest"
        return 0
    fi

    ###########################################################################
    # Deterministic sampling
    ###########################################################################

    python3 \
        - "$class_tmp" "$class_manifest" "$selected_count" "$SEED" "$class" <<'PY'

import sys
import random

input_file = sys.argv[1]
output_file = sys.argv[2]
n = int(sys.argv[3])
seed = int(sys.argv[4])
class_name = sys.argv[5]

with open(
    input_file,
    "r",
    encoding="utf-8",
    errors="surrogateescape"
) as f:

    relative_paths = [
        line.rstrip("\n")
        for line in f
        if line.rstrip("\n")
    ]

if n > len(relative_paths):
    raise RuntimeError(
        f"Class {class_name}: requested {n}, "
        f"but only {len(relative_paths)} images exist"
    )

###############################################################################
# Class-specific deterministic random generator
###############################################################################

rng = random.Random(
    f"{seed}:{class_name}"
)

selected = rng.sample(
    relative_paths,
    n
)

###############################################################################
# Sort selected paths for a deterministic manifest
###############################################################################

selected.sort()

with open(
    output_file,
    "w",
    encoding="utf-8",
    errors="surrogateescape"
) as f:

    for relative_path in selected:
        f.write(
            f"{class_name}\t{relative_path}\n"
        )

PY

    ###########################################################################
    # Manifest-only mode
    ###########################################################################

    if [[ "$MODE" == "manifest" ]]; then
        return
    fi

    ###########################################################################
    # Create output files while preserving the complete relative path.
    ###########################################################################

    while IFS=$'\t' read -r selected_class relative_path; do

        source="$INPUT/$relative_path"
        destination="$OUTPUT/$relative_path"

        #######################################################################
        # This creates:
        #
        #     OUTPUT/class_name/
        #
        # if it doesn't already exist.
        #######################################################################

        mkdir -p "$(dirname "$destination")"

        if [[ "$MODE" == "symlink" ]]; then

            ln -s "$source" "$destination"

        elif [[ "$MODE" == "copy" ]]; then

            cp -- "$source" "$destination"

        fi

    done < "$class_manifest"
}

###############################################################################
# Export function and variables for GNU parallel
###############################################################################

export -f select_class

export INPUT
export OUTPUT
export MODE
export SEED
export TMPDIR

###############################################################################
# Step 6: Process classes
###############################################################################

echo
echo "Selecting images..."
echo

if [[ "$HAVE_PARALLEL" -eq 1 && "$WORKERS" -gt 1 ]]; then

    parallel \
        --colsep '\t' \
        --jobs "$WORKERS" \
        --no-notice \
        select_class {1} {2} {3} {4} \
        :::: "$TMP_CLASS_JOBS"

else

    while IFS=$'\t' read -r class count selected class_dir; do

        select_class \
            "$class" \
            "$count" \
            "$selected" \
            "$class_dir"

    done < "$TMP_CLASS_JOBS"

fi

###############################################################################
# Step 7: Build final manifest
###############################################################################

echo "Building manifest..."

if [[ "$MODE" == "manifest" ]]; then

    # Collect manifests generated by workers.
    find "$TMPDIR" \
        -maxdepth 1 \
        -type f \
        -name 'manifest_*.tsv' \
        -print0 |
    xargs -0 cat |
    sort -t $'\t' -k1,1 -k2,2 > "$OUTPUT_MANIFEST"

else

    find "$TMPDIR" \
        -maxdepth 1 \
        -type f \
        -name 'manifest_*.tsv' \
        -print0 |
    xargs -0 cat |
    sort -t $'\t' -k1,1 -k2,2 > "$OUTPUT_MANIFEST"

fi

###############################################################################
# Step 8: Verify manifest
###############################################################################

MANIFEST_COUNT=$(wc -l < "$OUTPUT_MANIFEST")

echo
echo "Manifest contains: $MANIFEST_COUNT images"

if [[ "$MANIFEST_COUNT" -ne "$TARGET" ]]; then

    echo
    echo "ERROR: manifest contains $MANIFEST_COUNT images."
    echo "Expected: $TARGET"
    echo

    exit 1
fi

###############################################################################
# Step 9: Verify class counts in manifest
###############################################################################

python3 - \
    "$OUTPUT_MANIFEST" \
    "$TMPDIR/manifest_class_counts.tsv" <<'PY'

import sys
from collections import Counter

manifest = sys.argv[1]
output = sys.argv[2]

counts = Counter()

with open(
    manifest,
    "r",
    encoding="utf-8",
    errors="surrogateescape"
) as f:

    for line in f:

        line = line.rstrip("\n")

        if not line:
            continue

        class_name, relative_path = line.split("\t", 1)

        counts[class_name] += 1

with open(
    output,
    "w",
    encoding="utf-8"
) as f:

    for class_name in sorted(counts):

        f.write(
            f"{class_name}\t{counts[class_name]}\n"
        )

PY

###############################################################################
# Step 10: Save metadata
###############################################################################

cat > "$DATASET_INFO" <<EOF
Operation:
Proportional random sampling

Original dataset:
$INPUT

Output dataset:
$OUTPUT

Original number of images:
$TOTAL_IMAGES

Selected number of images:
$TARGET

Number of classes:
$NUM_CLASSES

Random seed:
$SEED

Workers:
$WORKERS

Sampling method:
Class-proportional sampling using the largest-remainder method.

Random sampling:
Performed independently for each class.

Manifest:
$OUTPUT_MANIFEST

Manifest paths:
Relative to the input dataset root.

Reproduction command:

$0 \\
    --input "$INPUT" \\
    --output "$OUTPUT/reproduced" \\
    --manifest "$OUTPUT_MANIFEST" \\
    --mode symlink

The manifest is the authoritative definition of the selected dataset.
EOF

###############################################################################
# Final output
###############################################################################

echo
echo "============================================================"
echo "Finished"
echo "============================================================"
echo "Original images : $TOTAL_IMAGES"
echo "Selected images : $MANIFEST_COUNT"
echo "Classes         : $NUM_CLASSES"
echo "Seed            : $SEED"
echo "Workers         : $WORKERS"
echo
echo "Manifest:"
echo "  $OUTPUT_MANIFEST"
echo
echo "Class counts:"
echo "  $CLASS_COUNTS"
echo
echo "Class allocations:"
echo "  $CLASS_ALLOCATIONS"
echo
echo "Metadata:"
echo "  $DATASET_INFO"
echo "============================================================"
echo
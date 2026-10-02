#!/usr/bin/env bash

set -u

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 SOURCE_DIR DEST_DIR"
    exit 1
fi

SOURCE_DIR="$(realpath "$1")"
DEST_DIR="$(realpath -m "$2")"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: source directory does not exist: $SOURCE_DIR"
    exit 1
fi

if [ "$SOURCE_DIR" = "$DEST_DIR" ]; then
    echo "Error: SOURCE_DIR and DEST_DIR must be different."
    exit 1
fi

mkdir -p "$DEST_DIR"

export DEST_DIR

cd "$SOURCE_DIR"

find . -type f -iname '*.png' -print0 |
parallel -0 --jobs 32 '
    src="{}"
    rel="${src#./}"
    dst="$DEST_DIR/$rel"

    mkdir -p "$(dirname "$dst")"

    tmp="${dst}.tmp"

    if convert "$src" \
        -background white \
        -alpha remove \
        -alpha off \
        -colorspace sRGB \
        -type TrueColor \
        "$tmp"
    then
        mv "$tmp" "$dst"
    else
        echo "CORRUPTED: $src"
        rm -f "$tmp" "$dst"
    fi
'
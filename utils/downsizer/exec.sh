#!/bin/bash

ENV_PATH=/path/to/env
source $ENV_PATH/bin/activate

./downsize_dataset.sh \
    --input input/path \
    --output output/path \
    --num-images 1000000 \
    --seed 42 \
    --workers 16 \
    --mode symlink
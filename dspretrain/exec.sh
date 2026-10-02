#!/bin/bash

# Activate virtualenv
ENV_PATH=/path/to/env
source $ENV_PATH/bin/activate

# Master port
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# model size
MODEL=tiny
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=deadleaves1M
# num of classes
CLASSES=1000
# num of epochs
EPOCHS=300
# path to train dataset
SOURCE_DATASET=/path/to/dataset
# output dir path
OUT_DIR=./output/train
# num of GPUs
NGPUS=2
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=512
# preprocess module
PREPROCESS=resize

PYTHONUNBUFFERED=1
PYTHONWARNINGS="ignore" 

torchrun --nproc_per_node=$NGPUS --master_port=$MASTER_PORT dspretrain.py ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment ${DATA_NAME}_${PREPROCESS} \
    --input-size 3 224 224 --preprocess $PREPROCESS \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --min-lr 1.0e-5 --warmup-lr 1.0e-6 --warmup-iter 5000 --cooldown-epochs 0 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 8 --pin-mem --eval-metric loss \
    --output ${OUT_DIR} \
    --amp &> progress.txt
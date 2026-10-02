#!/bin/bash

# Activate virtualenv
ENV_PATH=/path/to/env
source $ENV_PATH/bin/activate

# Master port
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

## Parameters ##

# Dataset
DATASET_PARAMS="../config/classes.csv"
DATASET_CFG="../config/classes.cfg"
EXPNAME="test"
DATASET_SIZE=150000640
RES=224
KERNEL_RES=512
MIN_RAND_KERNEL_RES=0

# Model Parameters
MODEL="deit_tiny_patch16_224"
NUM_CLASSES=1000

# Training parameters
WARMUP_ITERS=5000
LR=0.001
OPT="adamw"
WEIGHT_DECAY=0.05
BATCH_SIZE=512
SCHED="cosine"

# Data augmentation parameters
NUM_OPS=2
MAGNITUDE=28
AUG_REPEATS=2
MIXUP=0.8
CUTMIX=1.0 
REPROB=0.25
DROP_PATH=0.1
SMOOTHING=0.1

# Pretrain
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT pretrain.py \
    --dataset-csv-path $DATASET_PARAMS --dataset-size $DATASET_SIZE \
    --res $RES --kernel-res $KERNEL_RES --min-rand-kernel-res $MIN_RAND_KERNEL_RES --prep resize \
    --kernel-file ../src/cuda/batch/kernels/vatom_simple_batch.cuh --experiment $EXPNAME \
    --model $MODEL --num-classes $NUM_CLASSES --amp --log-interval 100 \
    --warmup-iters $WARMUP_ITERS --lr $LR --opt $OPT --weight-decay $WEIGHT_DECAY -b $BATCH_SIZE --sched $SCHED \
    --num-ops $NUM_OPS --magnitude $MAGNITUDE --aug-repeats $AUG_REPEATS \
    --mixup $MIXUP --cutmix $CUTMIX --reprob $REPROB --drop-path $DROP_PATH --smoothing $SMOOTHING &> progress.txt
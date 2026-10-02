#!/bin/bash

# Activate virtualenv
ENV_PATH=/path/to/env
source $ENV_PATH/bin/activate

# Workers
DATASET_WORKERS=8

# Master port
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Paths
DATADIR=/data/dir
PROJECTDIR=/path/to/clurred_course_separability
EXP=deadleaves1M_resize

# Model
MODEL="deit_tiny_patch16_224"
PRETRAIN_PATH=/path/to/model.pth.tar

# Learning parameters
EPOCHS=1000
WARMUP_EPOCHS=10
LR=0.015
OPT="sgd"
WEIGHT_DECAY=0.0001
BATCH_SIZE=384
SCHED="cosine"

# Data augmentation
AA="rand-m9-mstd0.5-inc1"
MIXUP=0.8
CUTMIX=1.0
REPROB=0.25

# Regularizers
DROP_PATH=0.1
SMOOTHING=0.1

# CIFAR100
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT finetune.py \
 --data-dir $DATADIR/cifar100 --dataset torch/cifar100 --val-split test --num-classes 100 -j $DATASET_WORKERS \
 --model $MODEL --pretrained --pretrained-path $PRETRAIN_PATH \
 --epochs $EPOCHS --warmup-epochs $WARMUP_EPOCHS \
 --lr $LR --opt $OPT --weight-decay $WEIGHT_DECAY -b $BATCH_SIZE --sched $SCHED \
 --aa $AA --mixup $MIXUP --cutmix $CUTMIX --reprob $REPROB --crop-pct 1.0 \
 --drop-path $DROP_PATH --smoothing $SMOOTHING --amp &> results/datasets/$EXP/progress_c100.txt

# VOC12
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT finetune.py \
 --data-dir $DATADIR/voc12 --val-split val --num-classes 20 -j $DATASET_WORKERS \
 --model $MODEL --pretrained --pretrained-path $PRETRAIN_PATH \
 --epochs $EPOCHS --warmup-epochs $WARMUP_EPOCHS \
 --lr $LR --opt $OPT --weight-decay $WEIGHT_DECAY -b $BATCH_SIZE --sched $SCHED \
 --aa $AA --mixup $MIXUP --cutmix $CUTMIX --reprob $REPROB --crop-pct 1.0 \
 --drop-path $DROP_PATH --smoothing $SMOOTHING --amp &> results/datasets/$EXP/progress_voc12.txt

# IMGNET100
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT finetune.py \
 --data-dir $DATADIR/imagenet100 --val-split val --num-classes 100 -j $DATASET_WORKERS \
 --model $MODEL --pretrained --pretrained-path $PRETRAIN_PATH \
 --epochs $EPOCHS --warmup-epochs $WARMUP_EPOCHS \
 --lr $LR --opt $OPT --weight-decay $WEIGHT_DECAY -b $BATCH_SIZE --sched $SCHED \
 --aa $AA --mixup $MIXUP --cutmix $CUTMIX --reprob $REPROB --crop-pct 1.0 \
 --drop-path $DROP_PATH --smoothing $SMOOTHING --amp &> results/datasets/$EXP/progress_imnet100.txt
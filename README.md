# Blurred Coarse Separability

Code repository for the JCAM CMMSE-submitted paper:

**Assessing Vision Pretraining Datasets Without Full-Scale Model Training**

## Requirements

### Python Environment

- **Python:** 3.11.9
- Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Preprocessing Tools

The preprocessing scripts require the following tools:

- **GNU Parallel:** Version dated 2026/07/22
- **ImageMagick:** 6.9.13-25 Q16 x86_64 (build 18639)

ImageMagick documentation: [https://legacy.imagemagick.org](https://legacy.imagemagick.org)

### GPU-Based Pretraining

GPU-based pretraining using the `pretrain/` directory requires **NVIDIA CUDA 12+** and the NVIDIA CUDA compiler (`nvcc`).

## Repository Structure

### `utils/`

Scripts for dataset preprocessing and preparation.

- **`downsizer/`** — Randomly selects a specified number of files from a dataset using a given random seed. In our experiments, we selected 1 million files.
- **`kubricproc/`** — Processes the Kubric ShapeNet dataset, originally distributed as TFRecord files. The scripts decompress the data into a class-based directory structure and convert RGBA images to RGB using GNU Parallel and ImageMagick.
- **`parquetfiles/`** — Converts Hugging Face datasets stored in Parquet format into a class-based directory structure.

### `pretrain/`

Scripts for GPU-generation-based pretraining on VisualAtoms.

### `config/`

Configuration files for GPU-generation-based pretraining.

### `dspretrain/`

Scripts for pretraining on datasets organized in a class-based directory structure. Adapted from [CVPR2023-FDSL-on-VisualAtom](https://github.com/masora1030/CVPR2023-FDSL-on-VisualAtom) with minor modifications.

### `finetune/`

Scripts for fine-tuning pretrained checkpoints on the following datasets:

- CIFAR-100
- VOC12
- ImageNet-100

Complete outputs from the different fine-tuning experiments are also available in this directory.

### `vendimi_metrics/`

Scripts for computing vendi and nmi metrics
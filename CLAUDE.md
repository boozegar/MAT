# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAT (Mask-Aware Transformer) is a research implementation for large hole image inpainting using transformers. This is a PyTorch-based project that won CVPR 2022 Best Paper Finalist (Oral). The codebase is built upon StyleGAN2-ADA and implements both training and inference for image inpainting tasks.

## Environment Setup

**Requirements:**
- Python 3.7
- PyTorch 1.7.1
- CUDA 11.0
- Dependencies listed in `requirements.txt`

**Installation:**
```bash
pip install -r requirements.txt
```

It's highly recommended to use Conda/MiniConda to avoid compilation errors.

## Key Commands

### Image Generation (Inference)
```bash
python generate_image.py --network model_path --dpath data_path --outdir out_path [--mpath mask_path]
```

Example:
```bash
python generate_image.py --network pretrained/CelebA-HQ.pkl --dpath test_sets/CelebA-HQ/images --mpath test_sets/CelebA-HQ/masks --outdir samples
```

### Training
```bash
python train.py \
    --outdir=output_path \
    --gpus=8 \
    --batch=32 \
    --metrics=fid36k5_full \
    --data=training_data_path \
    --data_val=val_data_path \
    --dataloader=datasets.dataset_512.ImageFolderMaskDataset \
    --mirror=True \
    --cond=False \
    --cfg=places512 \
    --aug=noaug \
    --generator=networks.mat.Generator \
    --discriminator=networks.mat.Discriminator \
    --loss=losses.loss.TwoStageLoss \
    --pr=0.1 \
    --pl=False \
    --truncation=0.5 \
    --style_mix=0.5 \
    --ema=10 \
    --lr=0.001
```

### Evaluation
Evaluation scripts are available in the `evaluatoin/` directory (note: typo in original folder name):
- `cal_fid_pids_uids.py` - FID, P-IDS, U-IDS metrics
- `cal_lpips.py` - LPIPS metric
- `cal_psnr_ssim_l1.py` - PSNR, SSIM, L1 metrics

## Architecture Overview

### Core Components

**Networks (`networks/`):**
- `mat.py` - Main MAT generator and discriminator implementation with transformer-based architecture
- `basic_module.py` - Basic building blocks (Conv2d layers, fully connected layers, etc.)

**Training (`training/`):**
- `training_loop.py` - Main training loop implementation
- `augment.py` - Data augmentation utilities

**Datasets (`datasets/`):**
- `dataset_256.py`, `dataset_512.py` - Dataset loaders for different resolutions
- `dataset_256_val.py`, `dataset_512_val.py` - Validation dataset loaders
- `mask_generator_*.py` - Random mask generation for training/testing

**Losses (`losses/`):**
- `loss.py` - Main loss functions including TwoStageLoss
- `pcp.py` - Perceptual loss implementation
- `vggNet.py` - VGG-based feature extraction

**Metrics (`metrics/`):**
- Various evaluation metrics implementations (FID, IS, KID, PPL, etc.)

### Key Architecture Notes

1. **Resolution Support**: The implementation supports 256x256 and 512x512 resolutions. Images must be multiples of 512 for the main model.

2. **Mask Format**: Masks use 0 for masked pixels and 1 for remaining pixels. When resizing images, pad masks with 0 values.

3. **Model Configurations**: 
   - CelebA-HQ models for faces
   - FFHQ models for high-quality faces  
   - Places365 models for natural scenes

4. **StyleGAN2 Foundation**: Built on StyleGAN2-ADA architecture with transformer modifications for mask-aware inpainting.

## Development Notes

- The project uses custom CUDA operations in `torch_utils/ops/` for optimized performance
- No formal test suite is present - evaluation is done through the metrics scripts
- Models are saved as `.pkl` files and loaded using the `legacy.py` module
- The codebase follows NVIDIA's StyleGAN2 code structure and conventions

## Pretrained Models

Models are available for download and should be placed in a `pretrained/` directory:
- CelebA-HQ (256x256 and 512x512)
- FFHQ (512x512) 
- Places365 (512x512, including full 8M image version)
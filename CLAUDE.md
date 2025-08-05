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

## Reference-Guided Inpainting Extension

The codebase has been extended to support reference image guidance for mural restoration and similar tasks:

### New Components

**Reference Encoder (`networks/reference_encoder.py`):**
- `ReferenceEncoder` - Extracts features from reference mural fragments
- `CrossAttentionFusion` - Fuses reference features with target image features via cross-attention
- `ReferenceStyleInjection` - Injects reference style into the generation process

**Enhanced Generator (`networks/mat_with_reference.py`):**
- `ReferenceGuidedSynthesisNet` - Extended synthesis network supporting reference guidance
- `ReferenceGuidedGenerator` - Modified generator with reference image input

**Enhanced Dataset (`datasets/dataset_with_reference.py`):**
- `ReferenceGuidedDataset` - Dataset loader supporting reference images during training

### Reference-Guided Commands

**Inference with Reference:**
```bash
python generate_with_reference.py \
    --network pretrained/model.pkl \
    --dpath input_images/ \
    --mpath masks/ \
    --rpath reference_murals/ \
    --outdir results/ \
    --ref_size 128 \
    --seeds 0-3
```

**Training with Reference:**
```bash
python train_with_reference.py \
    --outdir ./output \
    --data ./training_data \
    --reference_data ./reference_murals \
    --gpus 1 \
    --batch 8 \
    --ref_prob 0.8 \
    --ref_size 128 \
    --lr 0.001 \
    --epochs 100
```

### Reference Image Requirements

- **Format**: PNG, JPG, JPEG supported
- **Size**: Automatically resized to specified `ref_size` (default 128x128)
- **Content**: Should contain relevant mural patterns, textures, or style elements
- **Usage**: Reference images guide the inpainting process through cross-attention mechanisms

### Architecture Integration

1. **Feature Extraction**: Reference images are encoded using a lightweight CNN encoder
2. **Cross-Attention**: Target image features attend to reference features at multiple scales
3. **Style Injection**: Reference style features are injected into the generator's style codes
4. **Multi-Scale Fusion**: Integration happens at 16x16, 32x32, and 64x64 feature levels

## Pretrained Models

Models are available for download and should be placed in a `pretrained/` directory:
- CelebA-HQ (256x256 and 512x512)
- FFHQ (512x512) 
- Places365 (512x512, including full 8M image version)

Note: Reference-guided models require training with the new architecture and cannot directly use original pretrained weights for the reference modules.

## 4090 Single GPU Optimization

The codebase includes special optimizations for training on RTX 4090 (24GB VRAM):

### Lightweight Architecture (`networks/lightweight_reference.py`)

**Memory-Efficient Components:**
- `LightweightReferenceEncoder` - <5M parameters, 3-layer CNN encoder
- `ReferenceAdapter` - Minimal adapter module with multi-head attention (4 heads)
- `StyleInjector` - Lightweight style injection with learnable weights
- `MinimalReferenceModule` - Complete reference system <2M parameters

### Parameter Freezing Strategy (`networks/freezable_generator.py`)

**FreezableReferenceGenerator Features:**
- Freeze MAT backbone (reduces trainable params by ~95%)
- Selective layer unfreezing for fine-tuning
- Memory-optimized forward pass with gradient checkpointing
- Automatic parameter statistics and memory estimation

**Memory Management:**
```python
# Create optimized model
model = create_lightweight_model(pretrained_path, device)
model.freeze_mat_backbone()  # Only train reference modules

# Optional: unfreeze last few layers
model.unfreeze_selected_layers(['synthesis.to_style'])
```

### 4090 Training Commands

**Memory-Optimized Training:**
```bash
python train_4090_finetune.py \
    --pretrained_path pretrained/Places_512.pkl \
    --data_path ./training_data \
    --reference_path ./reference_murals \
    --output_dir ./finetune_output \
    --batch_size 2 \
    --accumulation_steps 4 \
    --epochs 50 \
    --lr_reference 1e-4 \
    --lr_backbone 1e-5
```

**Key 4090 Optimizations:**
- Batch size 2 with 4x gradient accumulation (effective batch size 8)
- Mixed precision training (FP16)
- Gradient checkpointing
- Memory-efficient data loading
- Automatic memory cleanup

**Optimized Inference:**
```bash
python infer_4090_optimized.py \
    --model_path ./finetune_output/final_model.pth \
    --pretrained_mat pretrained/Places_512.pkl \
    --input damaged_murals/ \
    --reference reference_patterns/ \
    --output restored_results/ \
    --seeds 0-2
```

### Memory Usage Summary

**Training Memory Breakdown:**
- MAT backbone (frozen): ~8GB parameter storage
- Reference modules (trainable): ~0.2GB parameters + gradients
- Activations (batch_size=2): ~6-8GB
- Peak usage: ~16-18GB (well within 24GB limit)

**Performance Metrics:**
- Training speed: ~2-3 sec/batch (RTX 4090)
- Inference speed: ~1-2 sec/image
- Convergence: Typically 20-50 epochs for good results

### Best Practices for 4090

1. **Start with frozen backbone**: Train only reference modules first
2. **Gradual unfreezing**: Selectively unfreeze layers if needed
3. **Monitor memory**: Use built-in memory monitoring
4. **Use mixed precision**: Automatic FP16 for speed and memory savings
5. **Batch accumulation**: Maintain effective batch size while fitting in memory
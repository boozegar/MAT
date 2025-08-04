import os
import re
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import cv2

import legacy
from networks.mat_with_reference import ReferenceGuidedGenerator
from datasets.mask_generator_512 import RandomMask


def load_reference_image(ref_path, size=128, device='cuda'):
    """加载并预处理参考图片"""
    try:
        image = np.array(PIL.Image.open(ref_path))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        # 转为RGB
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # 调整大小
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        
        # 归一化到[-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # HWC -> CHW -> BCHW
        image = torch.from_numpy(image.transpose([2, 0, 1])).unsqueeze(0).to(device)
        
        return image
        
    except Exception as e:
        print(f"Error loading reference image {ref_path}: {e}")
        return torch.zeros(1, 3, size, size).to(device)


def load_image(image_path, resolution=512, device='cuda'):
    """加载并预处理输入图片"""
    image = np.array(PIL.Image.open(image_path))
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # 调整到指定分辨率
    H, W = image.shape[:2]
    if H != resolution or W != resolution:
        # 保持长宽比的resize和pad
        scale = resolution / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
        
        # 计算padding
        pad_H = resolution - new_H
        pad_W = resolution - new_W
        top, bottom = pad_H // 2, pad_H - pad_H // 2
        left, right = pad_W // 2, pad_W - pad_W // 2
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    # 归一化
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image.transpose([2, 0, 1])).unsqueeze(0).to(device)
    
    return image


def load_mask(mask_path, resolution=512, device='cuda'):
    """加载mask"""
    if mask_path is None:
        # 生成随机mask
        mask_generator = RandomMask(resolution, hole_range=[0.2, 0.8])
        mask = mask_generator.sample()
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        mask = np.array(PIL.Image.open(mask_path).convert('L'))
        if mask.shape[0] != resolution or mask.shape[1] != resolution:
            mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        
        # 转换为0-1 float，其中0表示需要修复的区域
        mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    
    return mask


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='Input image directory or single image path', required=True)
@click.option('--mpath', help='Input mask directory or single mask path', default=None)
@click.option('--rpath', help='Reference image directory or single reference path', default=None)
@click.option('--outdir', help='Output directory', type=str, required=True)
@click.option('--resolution', help='Image resolution', type=int, default=512)
@click.option('--ref_size', help='Reference image size', type=int, default=128)
@click.option('--seeds', help='Random seeds (e.g. 1,2,5-10)', type=str, default='0-3')
@click.option('--truncation-psi', help='Truncation psi', type=float, default=1.0)
def generate_images_with_reference(
    network_pkl: str,
    dpath: str,
    mpath: str,
    rpath: str,
    outdir: str,
    resolution: int,
    ref_size: int,
    seeds: str,
    truncation_psi: float
):
    """使用参考图引导生成修复图像"""
    
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network(f)['G_ema'].to(device)
    
    # 如果原模型不支持参考图，创建新的ReferenceGuidedGenerator
    if not hasattr(G.synthesis, 'use_reference'):
        print("Converting to reference-guided generator...")
        ref_G = ReferenceGuidedGenerator(
            z_dim=G.z_dim,
            c_dim=G.c_dim, 
            w_dim=G.w_dim,
            img_resolution=G.img_resolution,
            img_channels=G.img_channels,
            use_reference=True
        ).to(device)
        
        # 复制权重（除了新增的参考图相关模块）
        ref_G.mapping.load_state_dict(G.mapping.state_dict())
        # 只复制原有的synthesis部分
        ref_G.synthesis.first_stage.load_state_dict(G.synthesis.first_stage.state_dict())
        ref_G.synthesis.enc.load_state_dict(G.synthesis.enc.state_dict())
        ref_G.synthesis.to_square.load_state_dict(G.synthesis.to_square.state_dict())
        ref_G.synthesis.to_style.load_state_dict(G.synthesis.to_style.state_dict())
        ref_G.synthesis.dec.load_state_dict(G.synthesis.dec.state_dict())
        
        G = ref_G
    
    os.makedirs(outdir, exist_ok=True)
    
    # 解析seeds
    seeds = [int(s) for s in re.sub(r'[,-]', ' ', seeds).split()]
    if len(seeds) == 2:
        seeds = list(range(seeds[0], seeds[1] + 1))
    
    # 获取输入文件列表
    if os.path.isfile(dpath):
        image_files = [dpath]
    else:
        image_files = [os.path.join(dpath, f) for f in os.listdir(dpath) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 获取参考图文件列表
    reference_files = []
    if rpath:
        if os.path.isfile(rpath):
            reference_files = [rpath]
        elif os.path.isdir(rpath):
            reference_files = [os.path.join(rpath, f) for f in os.listdir(rpath)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f'Found {len(image_files)} input images')
    print(f'Found {len(reference_files)} reference images')
    
    # 处理每个输入图像
    for img_idx, image_path in enumerate(image_files):
        print(f'Processing image {img_idx + 1}/{len(image_files)}: {os.path.basename(image_path)}')
        
        # 加载输入图像
        input_image = load_image(image_path, resolution, device)
        
        # 加载mask
        if mpath:
            if os.path.isfile(mpath):
                mask_path = mpath
            else:
                mask_name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                mask_path = os.path.join(mpath, mask_name)
                if not os.path.exists(mask_path):
                    mask_path = None
        else:
            mask_path = None
        
        mask = load_mask(mask_path, resolution, device)
        
        # 为每个seed生成结果
        for seed_idx, seed in enumerate(seeds):
            print(f'  Seed {seed}...')
            
            # 生成随机向量
            torch.manual_seed(seed)
            z = torch.randn(1, G.z_dim, device=device)
            c = torch.zeros(1, G.c_dim, device=device)
            
            # 根据是否有参考图决定处理方式
            if reference_files:
                # 随机选择参考图或使用对应的参考图
                if len(reference_files) == 1:
                    ref_path = reference_files[0]
                elif len(reference_files) == len(image_files):
                    ref_path = reference_files[img_idx]
                else:
                    ref_path = reference_files[img_idx % len(reference_files)]
                
                reference_image = load_reference_image(ref_path, ref_size, device)
                
                # 生成结果
                with torch.no_grad():
                    result = G(input_image, mask, z, c, reference_img=reference_image,
                             truncation_psi=truncation_psi, noise_mode='const')
                
                suffix = f"_ref_{os.path.splitext(os.path.basename(ref_path))[0]}_seed{seed:04d}.png"
            else:
                # 不使用参考图
                with torch.no_grad():
                    result = G(input_image, mask, z, c, reference_img=None,
                             truncation_psi=truncation_psi, noise_mode='const')
                
                suffix = f"_seed{seed:04d}.png"
            
            # 保存结果
            result_image = (result.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)
            result_image = np.clip(result_image, 0, 255)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(outdir, base_name + suffix)
            PIL.Image.fromarray(result_image).save(output_path)
            
            print(f'    Saved: {output_path}')
    
    print('Done!')


if __name__ == "__main__":
    generate_images_with_reference()
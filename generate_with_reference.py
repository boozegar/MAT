"""
参考图引导的MAT推理脚本
4090优化版本 - 支持内存优化和批处理
"""
import os
import re
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import cv2
import time
from tqdm import tqdm
import glob

import legacy
from networks.mat_with_reference import create_reference_model
from datasets.mask_generator_512 import RandomMask


def load_and_preprocess_image(image_path, resolution=512, device='cuda'):
    """高效的图像加载和预处理"""
    try:
        # 使用cv2加载，更快
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 快速resize
        if image.shape[0] != resolution or image.shape[1] != resolution:
            image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
        
        # 转换为tensor并移到GPU
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        image = (image / 127.5 - 1.0).unsqueeze(0).to(device, non_blocking=True)
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return torch.zeros(1, 3, resolution, resolution, device=device)


def load_mask(mask_path, resolution=512, device='cuda'):
    """加载mask"""
    if mask_path is None or not os.path.exists(mask_path):
        # 生成随机mask
        mask_generator = RandomMask(resolution, hole_range=[0.2, 0.8])
        mask = mask_generator.sample()
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape[0] != resolution or mask.shape[1] != resolution:
                mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
            
            # 转换为0-1 float，其中0表示需要修复的区域
            mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        except:
            # 如果加载失败，生成随机mask
            mask_generator = RandomMask(resolution, hole_range=[0.2, 0.8])
            mask = mask_generator.sample()
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    
    return mask


def load_reference_image(ref_path, size=128, device='cuda'):
    """加载参考图"""
    if ref_path is None or not os.path.exists(ref_path):
        return torch.zeros(1, 3, size, size, device=device)
    
    try:
        image = cv2.imread(ref_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        image = (image / 127.5 - 1.0).unsqueeze(0).to(device, non_blocking=True)
        
        return image
    except:
        return torch.zeros(1, 3, size, size, device=device)


def save_result(tensor_image, save_path):
    """快速保存结果"""
    image = tensor_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image = np.clip((image * 127.5 + 127.5), 0, 255).astype(np.uint8)
    
    # 使用cv2保存，更快
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)


class OptimizedInference:
    """
    4090优化的推理引擎
    """
    def __init__(self, model_path, pretrained_mat_path=None, device='cuda', 
                 mixed_precision=True, lightweight_mode=False):
        self.device = device
        self.mixed_precision = mixed_precision
        self.lightweight_mode = lightweight_mode
        
        print(f"加载模型: {model_path}")
        
        # 加载模型
        if model_path.endswith('.pth'):
            # 加载微调后的检查点
            if pretrained_mat_path is None:
                raise ValueError("使用.pth检查点时需要提供预训练MAT模型路径")
            
            checkpoint = torch.load(model_path, map_location=device)
            self.lightweight_mode = checkpoint.get('lightweight_mode', lightweight_mode)
            
            self.model = create_reference_model(
                pretrained_mat_path, device, self.lightweight_mode
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从检查点加载完成 (轻量级模式: {self.lightweight_mode})")
        else:
            # 直接加载预训练模型
            self.model = create_reference_model(model_path, device, lightweight_mode)
        
        # 设置为评估模式
        self.model.eval()
        
        # 优化设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        print("模型加载完成")
    
    @torch.no_grad()
    def infer_single(self, image_path, mask_path=None, ref_path=None, 
                    resolution=512, ref_size=128, seed=0):
        """单张图像推理"""
        # 设置随机种子
        torch.manual_seed(seed)
        
        # 加载图像
        image = load_and_preprocess_image(image_path, resolution, self.device)
        mask = load_mask(mask_path, resolution, self.device)
        reference = load_reference_image(ref_path, ref_size, self.device)
        
        # 生成随机向量
        z = torch.randn(1, 512, device=self.device)
        c = torch.zeros(1, 0, device=self.device)
        
        # 推理
        if self.mixed_precision:
            with autocast():
                result = self.model(image, mask, z, c, reference_img=reference,
                                  truncation_psi=1.0, noise_mode='const')
        else:
            result = self.model(image, mask, z, c, reference_img=reference,
                              truncation_psi=1.0, noise_mode='const')
        
        return result
    
    def batch_infer(self, image_list, mask_list=None, ref_list=None,
                   output_dir=None, resolution=512, ref_size=128, seeds=[0]):
        """批量推理"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        total_time = 0
        
        for i, image_path in enumerate(tqdm(image_list, desc="推理中")):
            start_time = time.time()
            
            # 获取对应的mask和参考图
            mask_path = mask_list[i] if mask_list and i < len(mask_list) else None
            ref_path = ref_list[i] if ref_list and i < len(ref_list) else None
            
            for seed in seeds:
                # 推理
                result = self.infer_single(
                    image_path, mask_path, ref_path,
                    resolution, ref_size, seed
                )
                
                # 保存结果
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    if len(seeds) > 1:
                        output_name = f"{base_name}_seed{seed:03d}.png"
                    else:
                        output_name = f"{base_name}_restored.png"
                    
                    # 添加参考图信息到文件名
                    if ref_path:
                        ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                        output_name = f"{base_name}_ref_{ref_name}_seed{seed:03d}.png"
                    
                    output_path = os.path.join(output_dir, output_name)
                    save_result(result, output_path)
                
                results.append(result)
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 定期清理显存
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_time = total_time / len(image_list)
        print(f"平均推理时间: {avg_time:.2f}秒/图")
        
        return results


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename or checkpoint path', required=True)
@click.option('--pretrained_mat', help='Pretrained MAT path (required when using .pth checkpoint)')
@click.option('--dpath', help='Input image directory or single image path', required=True)
@click.option('--mpath', help='Input mask directory or single mask path', default=None)
@click.option('--rpath', help='Reference image directory or single reference path', default=None)
@click.option('--outdir', help='Output directory', type=str, required=True)
@click.option('--resolution', help='Image resolution', type=int, default=512)
@click.option('--ref_size', help='Reference image size', type=int, default=128)
@click.option('--seeds', help='Random seeds (e.g. 1,2,5-10)', type=str, default='0-3')
@click.option('--truncation-psi', help='Truncation psi', type=float, default=1.0)
@click.option('--lightweight', is_flag=True, help='Use lightweight mode for 4090')
@click.option('--mixed_precision', is_flag=True, default=True, help='Use mixed precision')
def generate_images_with_reference(
    network_pkl: str,
    pretrained_mat: str,
    dpath: str,
    mpath: str,
    rpath: str,
    outdir: str,
    resolution: int,
    ref_size: int,
    seeds: str,
    truncation_psi: float,
    lightweight: bool,
    mixed_precision: bool
):
    """使用参考图引导生成修复图像 - 4090优化版本"""
    
    print('=' * 60)
    print('参考图引导MAT推理')
    if lightweight:
        print('4090轻量级优化模式')
    print('=' * 60)
    
    # 检查模型路径
    if network_pkl.endswith('.pth') and not pretrained_mat:
        print("错误: 使用.pth检查点时需要提供--pretrained_mat参数")
        return
    
    # 解析种子
    if '-' in seeds:
        start, end = map(int, seeds.split('-'))
        seed_list = list(range(start, end + 1))
    else:
        seed_list = [int(s) for s in seeds.split(',')]
    
    print(f"使用种子: {seed_list}")
    
    # 创建推理引擎
    inference_engine = OptimizedInference(
        model_path=network_pkl,
        pretrained_mat_path=pretrained_mat,
        lightweight_mode=lightweight,
        mixed_precision=mixed_precision
    )
    
    # 准备输入文件列表
    if os.path.isfile(dpath):
        image_list = [dpath]
        mask_list = [mpath] if mpath else None
        ref_list = [rpath] if rpath else None
    else:
        # 目录模式
        image_list = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_list.extend(glob.glob(os.path.join(dpath, ext)))
        image_list.sort()
        
        mask_list = None
        if mpath and os.path.isdir(mpath):
            mask_list = []
            for img_path in image_list:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(mpath, f"{img_name}.png")
                mask_list.append(mask_path if os.path.exists(mask_path) else None)
        
        ref_list = None
        if rpath:
            if os.path.isdir(rpath):
                ref_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    ref_files.extend(glob.glob(os.path.join(rpath, ext)))
                if ref_files:
                    # 循环使用参考图
                    ref_list = (ref_files * ((len(image_list) // len(ref_files)) + 1))[:len(image_list)]
                else:
                    print(f"警告: 在 {rpath} 中没有找到参考图")
            else:
                ref_list = [rpath] * len(image_list)
    
    print(f"找到 {len(image_list)} 张图像进行处理")
    if ref_list:
        print(f"使用 {len(set(ref_list))} 张不同的参考图")
    
    # 执行推理
    start_time = time.time()
    results = inference_engine.batch_infer(
        image_list, mask_list, ref_list, outdir,
        resolution, ref_size, seed_list
    )
    total_time = time.time() - start_time
    
    print(f"处理完成!")
    print(f"总时间: {total_time:.1f}秒")
    print(f"生成图像: {len(results)} 张")
    print(f"输出目录: {outdir}")


if __name__ == "__main__":
    # 使用示例
    print("""
    4090优化推理示例:
    
    # 使用微调后的模型
    python generate_with_reference.py \\
        --network ./output/final_model.pth \\
        --pretrained_mat pretrained/Places_512.pkl \\
        --dpath damaged_murals/ \\
        --rpath reference_patterns/ \\
        --outdir restored_results/ \\
        --lightweight \\
        --seeds 0-2
    
    # 使用原始预训练模型
    python generate_with_reference.py \\
        --network pretrained/Places_512.pkl \\
        --dpath damaged_murals/ \\
        --rpath reference_patterns/ \\
        --outdir restored_results/ \\
        --seeds 0-2
    """)
    
    generate_images_with_reference()
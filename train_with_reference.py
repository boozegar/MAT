"""
训练支持参考图引导的MAT模型
4090优化版本 - 支持参数冻结和内存优化
"""
import os
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from networks.mat_with_reference import create_reference_model
from datasets.dataset_with_reference import create_reference_guided_dataset


class MemoryOptimizedTrainer:
    """
    内存优化的训练器 - 4090专用
    """
    def __init__(self, 
                 model,
                 device='cuda',
                 batch_size=2,  # 4090建议batch_size=2
                 accumulation_steps=4,  # 梯度累积，有效batch_size=8
                 mixed_precision=True,
                 gradient_checkpointing=True):
        
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        
        # 混合精度训练
        self.scaler = GradScaler() if mixed_precision else None
        
        # 启用梯度检查点
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
            
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        print("4090优化训练器初始化完成")
        self.print_parameter_summary()
    
    def _enable_gradient_checkpointing(self):
        """启用梯度检查点以节省显存"""
        def checkpoint_wrapper(module):
            def forward_wrapper(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(module.forward, *args, **kwargs)
            return forward_wrapper
        
        # 对参考图模块启用检查点
        if hasattr(self.model.synthesis, 'reference_encoder'):
            for name, module in self.model.synthesis.reference_encoder.named_modules():
                if len(list(module.children())) == 0:  # 叶子模块
                    module.forward = checkpoint_wrapper(module)
    
    def print_parameter_summary(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print("=" * 50)
        print("参数统计:")
        print(f"  总参数量:     {total_params:,}")
        print(f"  可训练参数:   {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  冻结参数:     {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print("=" * 50)
        
        # 显存估算 (粗略)
        param_memory = total_params * 4 / (1024**3)  # float32, GB
        gradient_memory = trainable_params * 4 / (1024**3)
        
        print(f"预估显存使用:")
        print(f"  参数存储:     {param_memory:.1f} GB")
        print(f"  梯度存储:     {gradient_memory:.1f} GB")
        print(f"  总计(不含激活): {param_memory + gradient_memory:.1f} GB")
        print("=" * 50)
    
    def setup_optimizer(self, lr_reference=1e-4, lr_backbone=1e-5):
        """设置优化器 - 不同学习率"""
        reference_params = self.model.get_reference_parameters()
        backbone_params = self.model.get_backbone_parameters()
        
        param_groups = []
        if reference_params:
            param_groups.append({'params': reference_params, 'lr': lr_reference, 'name': 'reference'})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'})
        
        if not param_groups:
            raise ValueError("没有可训练的参数!")
        
        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        
        print(f"优化器设置完成，参数组数量: {len(param_groups)}")
        for i, group in enumerate(param_groups):
            print(f"  组{i} ({group.get('name', 'unknown')}): {len(group['params'])} 参数, lr={group['lr']}")
    
    def compute_losses(self, fake_images, real_images, masks):
        """计算损失"""
        # 只在mask区域计算损失
        mask_area = 1 - masks
        
        # L1重建损失
        l1_loss = self.l1_loss(fake_images * mask_area, real_images * mask_area)
        
        # L2损失（稳定训练）
        l2_loss = self.l2_loss(fake_images * mask_area, real_images * mask_area)
        
        # 整体图像的轻微约束
        global_l1 = self.l1_loss(fake_images, real_images) * 0.1
        
        total_loss = l1_loss + 0.1 * l2_loss + global_l1
        
        return {
            'total': total_loss,
            'l1': l1_loss,
            'l2': l2_loss,
            'global': global_l1
        }
    
    def train_step(self, batch_data):
        """单步训练"""
        if len(batch_data) == 4:
            images, masks, reference_imgs, _ = batch_data
        else:
            images, masks, reference_imgs = batch_data
        
        images = images.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        reference_imgs = reference_imgs.to(self.device, non_blocking=True)
        
        batch_size = images.shape[0]
        
        # 生成随机向量
        z = torch.randn(batch_size, 512, device=self.device)
        c = torch.zeros(batch_size, 0, device=self.device)
        
        # 前向传播
        if self.mixed_precision:
            with autocast():
                fake_images = self.model(images, masks, z, c, reference_img=reference_imgs)
                losses = self.compute_losses(fake_images, images, masks)
        else:
            fake_images = self.model(images, masks, z, c, reference_img=reference_imgs)
            losses = self.compute_losses(fake_images, images, masks)
        
        return losses, fake_images
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {'total': 0, 'l1': 0, 'l2': 0, 'global': 0}
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 训练步骤
            losses, fake_images = self.train_step(batch_data)
            
            # 梯度缩放和累积
            loss = losses['total'] / self.accumulation_steps
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 更新统计
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
            
            # 梯度累积完成后更新参数
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # 更新进度条
            avg_loss = total_losses['total'] / num_batches
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'mem': f'{torch.cuda.memory_reserved()/1024**3:.1f}GB'
            })
            
            # 定期清理显存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= num_batches
            
        return total_losses
    
    def save_checkpoint(self, epoch, save_path, losses=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'lightweight_mode': getattr(self.model, 'lightweight_mode', False)
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"检查点已保存: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"检查点已加载: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint.get('losses', {})


@click.command()
@click.option('--pretrained_path', required=True, help='预训练MAT模型路径')
@click.option('--data_path', required=True, help='训练数据路径')
@click.option('--reference_path', required=True, help='参考图数据路径')
@click.option('--output_dir', required=True, help='输出目录')
@click.option('--lightweight', is_flag=True, help='使用4090轻量级模式')
@click.option('--batch_size', default=2, help='批大小')
@click.option('--accumulation_steps', default=4, help='梯度累积步数')
@click.option('--epochs', default=50, help='训练轮数')
@click.option('--lr_reference', default=1e-4, help='参考图模块学习率')
@click.option('--lr_backbone', default=1e-5, help='主干网络学习率')
@click.option('--ref_prob', default=0.8, help='使用参考图的概率')
@click.option('--save_every', default=5, help='保存间隔')
@click.option('--resume', default=None, help='恢复训练的检查点')
@click.option('--unfreeze_layers', default=None, help='解冻的层名，用逗号分隔')
def main(pretrained_path, data_path, reference_path, output_dir, lightweight,
         batch_size, accumulation_steps, epochs, lr_reference, lr_backbone,
         ref_prob, save_every, resume, unfreeze_layers):
    """
    4090参考图引导MAT训练主函数
    """
    print("=" * 60)
    print("参考图引导MAT训练")
    if lightweight:
        print("4090轻量级优化模式")
    print("=" * 60)
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型
    print("\n创建模型...")
    model = create_reference_model(
        pretrained_path=pretrained_path, 
        device=device,
        lightweight_mode=lightweight
    )
    
    # 选择性解冻层
    if unfreeze_layers:
        layers_list = [layer.strip() for layer in unfreeze_layers.split(',')]
        model.unfreeze_selected_layers(layers_list)
    
    # 创建数据集
    print("\n创建数据集...")
    dataset = create_reference_guided_dataset(
        data_path=data_path,
        reference_path=reference_path,
        reference_prob=ref_prob,
        reference_size=128,
        resolution=512,
        hole_range=[0.2, 0.8]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"有效批大小: {batch_size * accumulation_steps}")
    
    # 创建训练器
    trainer = MemoryOptimizedTrainer(
        model=model,
        device=device,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        mixed_precision=True,
        gradient_checkpointing=lightweight  # 轻量级模式启用梯度检查点
    )
    
    trainer.setup_optimizer(lr_reference, lr_backbone)
    
    # 恢复训练
    start_epoch = 0
    if resume:
        start_epoch, _ = trainer.load_checkpoint(resume)
        start_epoch += 1
    
    # 训练循环
    print(f"\n开始训练 (从epoch {start_epoch})...")
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch}/{epochs-1}")
        
        # 训练
        losses = trainer.train_epoch(dataloader, epoch)
        
        # 打印损失
        print(f"Losses - Total: {losses['total']:.4f}, "
              f"L1: {losses['l1']:.4f}, L2: {losses['l2']:.4f}, "
              f"Global: {losses['global']:.4f}")
        
        # 保存检查点
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            trainer.save_checkpoint(epoch, checkpoint_path, losses)
        
        # 显存清理
        torch.cuda.empty_cache()
    
    # 保存最终模型
    final_path = os.path.join(output_dir, 'final_model.pth')
    trainer.save_checkpoint(epochs-1, final_path)
    
    print("\n训练完成!")
    print(f"最终模型已保存: {final_path}")


if __name__ == "__main__":
    # 设置环境变量优化性能
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # 示例使用说明
    print("""
    4090优化训练示例:
    
    # 轻量级模式 (推荐)
    python train_with_reference.py \\
        --pretrained_path pretrained/Places_512.pkl \\
        --data_path ./training_data \\
        --reference_path ./reference_murals \\
        --output_dir ./output \\
        --lightweight \\
        --batch_size 2 \\
        --accumulation_steps 4 \\
        --epochs 50
    
    # 全功能模式 (需要更多显存)
    python train_with_reference.py \\
        --pretrained_path pretrained/Places_512.pkl \\
        --data_path ./training_data \\
        --reference_path ./reference_murals \\
        --output_dir ./output \\
        --batch_size 1 \\
        --accumulation_steps 8 \\
        --epochs 50
    """)
    
    main()
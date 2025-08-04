"""
训练支持参考图引导的MAT模型的示例配置和启动脚本
"""
import os
import click
import torch
from training import training_loop
from networks.mat_with_reference import ReferenceGuidedGenerator
from datasets.dataset_with_reference import create_reference_guided_dataset


@click.command()
@click.option('--outdir', help='输出目录', required=True)
@click.option('--data', help='训练数据路径', required=True)
@click.option('--reference_data', help='参考图数据路径', default=None)
@click.option('--data_val', help='验证数据路径', default=None)
@click.option('--gpus', help='GPU数量', type=int, default=1)
@click.option('--batch', help='批大小', type=int, default=16)
@click.option('--resolution', help='图像分辨率', type=int, default=512)
@click.option('--ref_prob', help='使用参考图的概率', type=float, default=0.8)
@click.option('--ref_size', help='参考图尺寸', type=int, default=128)
@click.option('--resume', help='恢复训练的checkpoint路径', default=None)
@click.option('--lr', help='学习率', type=float, default=0.001)
@click.option('--epochs', help='训练轮数', type=int, default=100)
def train_reference_guided_mat(
    outdir: str,
    data: str, 
    reference_data: str,
    data_val: str,
    gpus: int,
    batch: int,
    resolution: int,
    ref_prob: float,
    ref_size: int,
    resume: str,
    lr: float,
    epochs: int
):
    """训练支持参考图引导的MAT模型"""
    
    print("=== 训练参考图引导的MAT模型 ===")
    print(f"输出目录: {outdir}")
    print(f"训练数据: {data}")
    print(f"参考图数据: {reference_data}")
    print(f"GPU数量: {gpus}")
    print(f"批大小: {batch}")
    print(f"参考图使用概率: {ref_prob}")
    print(f"参考图尺寸: {ref_size}")
    
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    
    # 配置参数
    training_options = {
        # 基本设置
        'outdir': outdir,
        'gpus': gpus,
        'batch_size': batch,
        'resolution': resolution,
        'epochs': epochs,
        
        # 数据设置
        'training_data_path': data,
        'reference_data_path': reference_data,
        'validation_data_path': data_val or data,
        'reference_prob': ref_prob,
        'reference_size': ref_size,
        
        # 网络设置
        'generator_class': ReferenceGuidedGenerator,
        'use_reference': reference_data is not None,
        
        # 训练设置
        'learning_rate': lr,
        'beta1': 0.0,
        'beta2': 0.99,
        'weight_decay': 0.0,
        
        # 损失权重
        'lambda_l1': 1.0,
        'lambda_perceptual': 0.1,
        'lambda_style': 120.0,
        'lambda_adv': 0.1,
        
        # 其他设置
        'resume_checkpoint': resume,
        'save_every': 1000,
        'log_every': 100,
        'eval_every': 2000,
        'sample_every': 500,
    }
    
    print("\n=== 开始训练 ===")
    
    # 这里应该调用修改后的training_loop
    # 由于原始training_loop比较复杂，这里提供一个简化的训练框架
    run_training(training_options)


def run_training(opts):
    """简化的训练循环框架"""
    
    print("初始化训练环境...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = create_reference_guided_dataset(
        data_path=opts['training_data_path'],
        reference_path=opts['reference_data_path'],
        reference_prob=opts['reference_prob'],
        reference_size=opts['reference_size'],
        resolution=opts['resolution'],
        hole_range=[0.2, 0.8]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建模型
    print("创建模型...")
    generator = opts['generator_class'](
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=opts['resolution'],
        img_channels=3,
        use_reference=opts['use_reference']
    ).to(device)
    
    # 创建判别器（这里使用原始的判别器）
    from networks.mat import Discriminator
    discriminator = Discriminator(
        c_dim=0,
        img_resolution=opts['resolution'],
        img_channels=3
    ).to(device)
    
    # 优化器
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=opts['learning_rate'],
        betas=(opts['beta1'], opts['beta2'])
    )
    
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=opts['learning_rate'],
        betas=(opts['beta1'], opts['beta2'])
    )
    
    # 损失函数
    l1_loss = torch.nn.L1Loss()
    
    print("开始训练循环...")
    
    # 这里是简化的训练循环，实际应该根据原始MAT的训练逻辑进行完整实现
    for epoch in range(opts['epochs']):
        generator.train()
        discriminator.train()
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 4:  # 有参考图
                images, masks, reference_imgs, _ = batch_data
            else:  # 无参考图
                images, masks, reference_imgs = batch_data
            
            images = images.to(device)
            masks = masks.to(device) 
            reference_imgs = reference_imgs.to(device)
            
            batch_size = images.shape[0]
            
            # 生成随机向量
            z = torch.randn(batch_size, 512, device=device)
            c = torch.zeros(batch_size, 0, device=device)
            
            # 生成器前向传播
            fake_images = generator(images, masks, z, c, 
                                  reference_img=reference_imgs if opts['use_reference'] else None)
            
            # 这里应该实现完整的GAN训练逻辑
            # 包括判别器损失、生成器损失、感知损失等
            
            # 简化的L1损失示例
            g_l1_loss = l1_loss(fake_images * (1 - masks), images * (1 - masks))
            
            g_optimizer.zero_grad()
            g_l1_loss.backward()
            g_optimizer.step()
            
            if batch_idx % opts['log_every'] == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, G_L1_Loss: {g_l1_loss.item():.4f}")
        
        print(f"Epoch {epoch} completed")
        
        # 保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(opts['outdir'], f'checkpoint_epoch_{epoch+1}.pkl')
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            print(f"保存checkpoint: {checkpoint_path}")
    
    print("训练完成!")


if __name__ == "__main__":
    # 示例使用方法说明
    print("""
    参考图引导MAT训练示例:
    
    python train_with_reference.py \\
        --outdir ./output \\
        --data ./training_data \\
        --reference_data ./reference_murals \\
        --gpus 1 \\
        --batch 8 \\
        --ref_prob 0.8 \\
        --ref_size 128 \\
        --lr 0.001 \\
        --epochs 100
    
    注意：这是一个简化的训练框架。完整的实现需要：
    1. 完整的GAN损失（对抗损失、感知损失、样式损失等）
    2. 判别器的训练逻辑  
    3. 学习率调度
    4. 验证和可视化
    5. 更完善的checkpoint管理
    """)
    
    train_reference_guided_mat()
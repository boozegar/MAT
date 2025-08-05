#!/bin/bash
# MAT参考图引导训练 - 新服务器部署脚本 (从第4步开始)

set -e

echo "========================================="
echo "MAT参考图引导训练 - 服务器部署脚本"
echo "注意: 请确保已手动克隆代码库并创建conda环境"
echo "========================================="

# 4. 安装依赖
echo "4. 安装PyTorch和依赖..."
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 5. 创建必要目录
echo "5. 创建目录结构..."
mkdir -p pretrained
mkdir -p data/training_images
mkdir -p data/reference_images
mkdir -p data/masks
mkdir -p output
mkdir -p test_images

# 6. 下载预训练模型提示
echo "6. 预训练模型下载说明:"
echo "   请手动下载预训练模型到 pretrained/ 目录:"
echo "   - Places_512_FullData.pkl (推荐用于壁画)"
echo "   - 下载链接: https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE"

echo ""
echo "========================================="
echo "部署完成! 接下来的步骤:"
echo "1. 下载预训练模型到 pretrained/ 目录"
echo "2. 准备训练数据到 data/ 目录"
echo "3. 运行: bash quick_start_training.sh"
echo "========================================="
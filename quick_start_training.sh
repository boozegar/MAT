#!/bin/bash
# 快速启动训练脚本 - 4090优化版本

set -e

echo "========================================="
echo "MAT参考图引导训练 - 4090快速启动"
echo "========================================="

# 检查必要文件
echo "1. 检查环境和文件..."

if [ ! -f "pretrained/Places_512_FullData.pkl" ] && [ ! -f "pretrained/Places_512.pkl" ]; then
    echo "❌ 错误: 未找到预训练模型文件"
    echo "请下载 Places_512_FullData.pkl 到 pretrained/ 目录"
    echo "下载链接: https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE"
    exit 1
fi

if [ ! -d "data/training_images" ] || [ -z "$(ls -A data/training_images)" ]; then
    echo "❌ 错误: 训练图像目录为空"
    echo "请将训练图像放到 data/training_images/ 目录"
    exit 1
fi

if [ ! -d "data/reference_images" ] || [ -z "$(ls -A data/reference_images)" ]; then
    echo "❌ 错误: 参考图像目录为空"
    echo "请将参考图像放到 data/reference_images/ 目录"
    exit 1
fi

# 确定使用的预训练模型
if [ -f "pretrained/Places_512_FullData.pkl" ]; then
    PRETRAINED_MODEL="pretrained/Places_512_FullData.pkl"
else
    PRETRAINED_MODEL="pretrained/Places_512.pkl"
fi

echo "✅ 使用预训练模型: $PRETRAINED_MODEL"

# 检查GPU
echo "2. 检查GPU状态..."
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo "检测到GPU: $GPU_NAME"

# 根据GPU选择配置
if [[ "$GPU_NAME" == *"4090"* ]]; then
    echo "✅ 检测到RTX 4090，使用轻量级配置"
    USE_LIGHTWEIGHT="--lightweight"
    BATCH_SIZE=2
    ACCUMULATION_STEPS=4
elif [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
    echo "✅ 检测到高端GPU，使用标准配置"
    USE_LIGHTWEIGHT=""
    BATCH_SIZE=4
    ACCUMULATION_STEPS=2
else
    echo "⚠️  未知GPU，使用保守配置"
    USE_LIGHTWEIGHT="--lightweight"
    BATCH_SIZE=1
    ACCUMULATION_STEPS=8
fi

# 创建输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/training_$TIMESTAMP"
mkdir -p $OUTPUT_DIR

echo "3. 开始训练..."
echo "   输出目录: $OUTPUT_DIR"
echo "   批大小: $BATCH_SIZE"
echo "   梯度累积: $ACCUMULATION_STEPS"
echo "   轻量级模式: ${USE_LIGHTWEIGHT:-否}"

# 启动训练
python train_with_reference.py \
    --pretrained_path "$PRETRAINED_MODEL" \
    --data_path data/training_images \
    --reference_path data/reference_images \
    --output_dir "$OUTPUT_DIR" \
    $USE_LIGHTWEIGHT \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --epochs 50 \
    --lr_reference 1e-4 \
    --lr_backbone 1e-5 \
    --ref_prob 0.8 \
    --save_every 5

echo ""
echo "========================================="
echo "训练完成! 模型已保存到: $OUTPUT_DIR"
echo "========================================="
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence
from networks.basic_module import FullyConnectedLayer, Conv2dLayer


@persistence.persistent_class
class ReferenceEncoder(nn.Module):
    """
    参考图编码器，用于提取参考壁画片段的特征
    """
    def __init__(self, 
                 in_channels=3,
                 feature_dim=512,
                 patch_size=16,
                 num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        
        # 卷积特征提取
        channels = [in_channels, 64, 128, 256, feature_dim]
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.conv_layers.append(
                Conv2dLayer(
                    in_channels=channels[i],
                    out_channels=channels[i+1], 
                    kernel_size=4,
                    down=2,
                    activation='lrelu'
                )
            )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征映射到不同尺度
        self.feature_mappers = nn.ModuleDict({
            'style': FullyConnectedLayer(feature_dim, feature_dim, activation='lrelu'),
            'local': FullyConnectedLayer(feature_dim, feature_dim, activation='lrelu'),
            'global': FullyConnectedLayer(feature_dim, feature_dim, activation='lrelu')
        })
        
    def forward(self, reference_img):
        """
        Args:
            reference_img: (B, C, H, W) 参考图像
        Returns:
            dict包含不同层级的特征表示
        """
        B = reference_img.shape[0]
        x = reference_img
        
        # 逐层提取特征
        features = []
        for conv in self.conv_layers:
            x = conv(x)
            features.append(x)
        
        # 全局特征
        global_feat = self.global_pool(x).view(B, -1)  # (B, feature_dim)
        
        # 局部特征（保持空间维度）
        local_feat = F.adaptive_avg_pool2d(x, (8, 8))  # (B, feature_dim, 8, 8)
        
        return {
            'global': self.feature_mappers['global'](global_feat),
            'local': local_feat,
            'style': self.feature_mappers['style'](global_feat),
            'raw_features': features
        }


@persistence.persistent_class 
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention模块，用于融合参考图特征和目标图像特征
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Query来自目标图像特征，Key和Value来自参考图像
        self.q = FullyConnectedLayer(dim, dim, bias=qkv_bias)
        self.kv_ref = FullyConnectedLayer(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = FullyConnectedLayer(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 用于融合的权重
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, target_feat, ref_feat, mask=None):
        """
        Args:
            target_feat: (B, N, C) 目标图像特征
            ref_feat: (B, M, C) 参考图像特征  
            mask: 可选的注意力mask
        """
        B, N, C = target_feat.shape
        _, M, _ = ref_feat.shape
        
        # Query来自目标图像
        q = self.q(target_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Key, Value来自参考图像
        kv = self.kv_ref(ref_feat).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, M, head_dim)
        
        # 计算attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用attention到value
        attended_feat = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        attended_feat = self.proj(attended_feat)
        attended_feat = self.proj_drop(attended_feat)
        
        # 融合原始特征和attended特征
        fused_feat = target_feat * (1 - self.fusion_weight) + attended_feat * self.fusion_weight
        
        return fused_feat, attn


@persistence.persistent_class
class ReferenceStyleInjection(nn.Module):
    """
    参考图样式注入模块，用于将参考图特征注入到style code中
    """
    def __init__(self, style_dim, ref_dim):
        super().__init__()
        self.style_mapper = FullyConnectedLayer(ref_dim, style_dim, activation='lrelu')
        self.fusion_layer = FullyConnectedLayer(style_dim * 2, style_dim, activation='lrelu')
        
    def forward(self, original_style, ref_features):
        """
        Args:
            original_style: (B, style_dim) 原始style code
            ref_features: (B, ref_dim) 参考图特征
        """
        ref_style = self.style_mapper(ref_features)
        combined_style = torch.cat([original_style, ref_style], dim=1)
        final_style = self.fusion_layer(combined_style)
        return final_style


# 4090优化版本 - 轻量级模块
@persistence.persistent_class
class LightweightReferenceEncoder(nn.Module):
    """
    轻量级参考图编码器，专为4090单卡设计
    参数量控制在5M以下
    """
    def __init__(self, 
                 in_channels=3,
                 feature_dim=256,  # 减小特征维度
                 num_layers=3):    # 减少层数
        super().__init__()
        self.feature_dim = feature_dim
        
        # 轻量级卷积编码器
        channels = [in_channels, 32, 64, feature_dim]
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 4, 2, 1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # 自适应池化 + 全连接
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)  # 输出4x4特征图
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 轻量级特征映射
        self.global_mapper = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
        # 局部特征处理
        self.local_conv = nn.Conv2d(feature_dim, feature_dim // 2, 1)
        
    def forward(self, reference_img):
        """
        Args:
            reference_img: (B, C, H, W) 参考图像，任意尺寸
        Returns:
            dict: 包含全局和局部特征
        """
        B = reference_img.shape[0]
        
        # 如果输入尺寸过大，先resize到合理大小
        if reference_img.shape[-1] > 256:
            reference_img = F.interpolate(reference_img, size=256, mode='bilinear', align_corners=False)
        
        x = reference_img
        
        # 卷积特征提取
        for conv in self.conv_layers:
            x = conv(x)
        
        # 全局特征
        global_feat = self.global_pool(x).view(B, -1)
        global_feat = self.global_mapper(global_feat)
        
        # 局部特征 (4x4)
        local_feat = self.adaptive_pool(x)
        local_feat = self.local_conv(local_feat)  # 降维减少计算
        
        return {
            'global': global_feat,        # (B, feature_dim)
            'local': local_feat,          # (B, feature_dim//2, 4, 4)
            'raw': x                      # 原始特征图
        }


@persistence.persistent_class 
class ReferenceAdapter(nn.Module):
    """
    参考图适配器，将参考图特征适配到MAT的特征空间
    这是唯一需要训练的模块，参数量极小
    """
    def __init__(self, mat_feature_dim, ref_feature_dim=256):
        super().__init__()
        
        # 特征维度适配
        self.feature_adapter = nn.Sequential(
            nn.Linear(ref_feature_dim, mat_feature_dim),
            nn.LayerNorm(mat_feature_dim),
            nn.GELU()
        )
        
        # 轻量级注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=mat_feature_dim,
            num_heads=4,  # 减少注意力头数
            batch_first=True,
            dropout=0.1
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(mat_feature_dim * 2, mat_feature_dim),
            nn.Sigmoid()
        )
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始化为小值，避免破坏预训练特征
        
    def forward(self, mat_features, ref_features):
        """
        Args:
            mat_features: (B, N, D) MAT的特征
            ref_features: (B, ref_dim) 参考图全局特征
        Returns:
            fused_features: (B, N, D) 融合后的特征
        """
        B, N, D = mat_features.shape
        
        # 适配参考图特征维度
        ref_adapted = self.feature_adapter(ref_features)  # (B, D)
        ref_adapted = ref_adapted.unsqueeze(1).expand(B, N, D)  # (B, N, D)
        
        # 自注意力融合（轻量级）
        attended_feat, _ = self.attention(mat_features, ref_adapted, ref_adapted)
        
        # 门控融合
        concat_feat = torch.cat([mat_features, attended_feat], dim=-1)
        gate_weight = self.gate(concat_feat)  # (B, N, D)
        
        # 加权融合
        fused = mat_features + self.alpha * gate_weight * attended_feat
        
        return fused


@persistence.persistent_class
class LightweightStyleInjector(nn.Module):
    """
    轻量级样式注入器，4090优化版本
    """
    def __init__(self, style_dim, ref_dim=256):
        super().__init__()
        
        self.style_adapter = nn.Sequential(
            nn.Linear(ref_dim, style_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim // 2, style_dim)
        )
        
        # 自适应权重
        self.weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, original_style, ref_global_feat):
        """
        Args:
            original_style: (B, style_dim) 原始style
            ref_global_feat: (B, ref_dim) 参考图全局特征
        """
        ref_style = self.style_adapter(ref_global_feat)
        return original_style + self.weight * ref_style
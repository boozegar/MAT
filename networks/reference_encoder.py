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
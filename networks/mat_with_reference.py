import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence
from networks.mat import Generator as OriginalGenerator, SynthesisNet, nf
from networks.basic_module import FullyConnectedLayer
from networks.reference_encoder import ReferenceEncoder, CrossAttentionFusion, ReferenceStyleInjection


@persistence.persistent_class
class ReferenceGuidedSynthesisNet(SynthesisNet):
    """
    支持参考图引导的SynthesisNet
    """
    def __init__(self, 
                 w_dim,
                 img_resolution,
                 img_channels=3,
                 use_reference=True,
                 ref_feature_dim=512,
                 **kwargs):
        super().__init__(w_dim, img_resolution, img_channels, **kwargs)
        
        self.use_reference = use_reference
        
        if use_reference:
            # 参考图编码器
            self.reference_encoder = ReferenceEncoder(
                in_channels=img_channels,
                feature_dim=ref_feature_dim
            )
            
            # Cross-attention融合模块（在不同层级）
            self.cross_attention_layers = nn.ModuleDict({
                '16': CrossAttentionFusion(dim=nf(4), num_heads=8),  # 16x16层
                '32': CrossAttentionFusion(dim=nf(5), num_heads=8),  # 32x32层
                '64': CrossAttentionFusion(dim=nf(6), num_heads=8),  # 64x64层
            })
            
            # 样式注入模块
            self.style_injection = ReferenceStyleInjection(
                style_dim=nf(2) * 2,  # 原始style维度
                ref_dim=ref_feature_dim
            )
    
    def forward(self, images_in, masks_in, ws, reference_img=None, noise_mode='random', return_stg1=False):
        """
        Args:
            images_in: 输入图像
            masks_in: 输入mask
            ws: style codes
            reference_img: 参考图像 (B, C, H, W)，可选
            noise_mode: 噪声模式
            return_stg1: 是否返回第一阶段结果
        """
        # 第一阶段（原始MAT）
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)
        
        # 编码器特征提取
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)
        
        # 处理16x16特征
        fea_16 = E_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        
        # 如果有参考图，进行特征融合
        if self.use_reference and reference_img is not None:
            # 提取参考图特征
            ref_features = self.reference_encoder(reference_img)
            
            # 在16x16层进行cross-attention融合
            B, C, H, W = fea_16.shape
            fea_16_flat = fea_16.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            
            # 参考图局部特征reshape用于attention
            ref_local = ref_features['local'].view(B, ref_features['local'].shape[1], -1).permute(0, 2, 1)  # (B, 64, C)
            
            # 应用cross-attention
            if '16' in self.cross_attention_layers:
                fused_feat, attn_map = self.cross_attention_layers['16'](fea_16_flat, ref_local)
                fea_16 = fused_feat.permute(0, 2, 1).view(B, C, H, W)
        
        E_features[4] = fea_16
        
        # 样式特征提取
        gs = self.to_style(fea_16)
        
        # 如果有参考图，融合样式特征
        if self.use_reference and reference_img is not None:
            gs = self.style_injection(gs, ref_features['style'])
        
        # 解码器
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)
        
        # 融合
        img = img * (1 - masks_in) + images_in * masks_in
        
        if not return_stg1:
            return img
        else:
            return img, out_stg1


@persistence.persistent_class  
class ReferenceGuidedGenerator(OriginalGenerator):
    """
    支持参考图引导的MAT Generator
    """
    def __init__(self,
                 z_dim,
                 c_dim,
                 w_dim,
                 img_resolution,
                 img_channels,
                 use_reference=True,
                 synthesis_kwargs={},
                 mapping_kwargs={}):
        
        # 更新synthesis_kwargs以使用我们的ReferenceGuidedSynthesisNet
        synthesis_kwargs.update({'use_reference': use_reference})
        
        super().__init__(z_dim, c_dim, w_dim, img_resolution, img_channels,
                        synthesis_kwargs, mapping_kwargs)
        
        # 替换synthesis网络
        self.synthesis = ReferenceGuidedSynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs
        )
        
    def forward(self, images_in, masks_in, z, c, reference_img=None,
                truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
                noise_mode='random', return_stg1=False):
        """
        Args:
            images_in: 输入图像
            masks_in: 输入mask  
            z: 潜在向量
            c: 条件向量
            reference_img: 参考图像 (B, C, H, W)，可选
            其他参数与原始Generator相同
        """
        ws = self.mapping(z, c, truncation_psi=truncation_psi, 
                         truncation_cutoff=truncation_cutoff,
                         skip_w_avg_update=skip_w_avg_update)
        
        if not return_stg1:
            img = self.synthesis(images_in, masks_in, ws, reference_img=reference_img, 
                               noise_mode=noise_mode)
            return img
        else:
            img, out_stg1 = self.synthesis(images_in, masks_in, ws, reference_img=reference_img,
                                         noise_mode=noise_mode, return_stg1=True)
            return img, out_stg1
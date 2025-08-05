import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence
from networks.mat import Generator as OriginalGenerator, SynthesisNet, nf
from networks.basic_module import FullyConnectedLayer
from networks.reference_encoder import (ReferenceEncoder, CrossAttentionFusion, ReferenceStyleInjection,
                                       LightweightReferenceEncoder, ReferenceAdapter, LightweightStyleInjector)


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
                 lightweight_mode=False,  # 4090优化模式
                 **kwargs):
        super().__init__(w_dim, img_resolution, img_channels, **kwargs)
        
        self.use_reference = use_reference
        self.lightweight_mode = lightweight_mode
        
        if use_reference:
            if lightweight_mode:
                # 4090优化版本 - 轻量级模块
                self.reference_encoder = LightweightReferenceEncoder(
                    in_channels=img_channels,
                    feature_dim=ref_feature_dim
                )
                
                # 只在16x16层融合以节省内存
                self.cross_attention_layers = nn.ModuleDict({
                    '16': ReferenceAdapter(nf(4), ref_feature_dim),
                })
                
                # 轻量级样式注入
                self.style_injection = LightweightStyleInjector(
                    style_dim=nf(2) * 2,
                    ref_dim=ref_feature_dim
                )
            else:
                # 原版全功能模块
                self.reference_encoder = ReferenceEncoder(
                    in_channels=img_channels,
                    feature_dim=ref_feature_dim
                )
                
                # Cross-attention融合模块（在不同层级）
                self.cross_attention_layers = nn.ModuleDict({
                    '16': CrossAttentionFusion(dim=nf(4), num_heads=8),
                    '32': CrossAttentionFusion(dim=nf(5), num_heads=8),
                    '64': CrossAttentionFusion(dim=nf(6), num_heads=8),
                })
                
                # 样式注入模块
                self.style_injection = ReferenceStyleInjection(
                    style_dim=nf(2) * 2,
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
            
            # 在16x16层进行融合
            B, C, H, W = fea_16.shape
            fea_16_flat = fea_16.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            
            if self.lightweight_mode:
                # 轻量级模式 - 直接适配器融合
                if '16' in self.cross_attention_layers:
                    fused_feat = self.cross_attention_layers['16'](fea_16_flat, ref_features['global'])
                    fea_16 = fused_feat.permute(0, 2, 1).view(B, C, H, W)
            else:
                # 全功能模式 - cross-attention融合
                ref_local = ref_features['local'].view(B, ref_features['local'].shape[1], -1).permute(0, 2, 1)
                if '16' in self.cross_attention_layers:
                    fused_feat, attn_map = self.cross_attention_layers['16'](fea_16_flat, ref_local)
                    fea_16 = fused_feat.permute(0, 2, 1).view(B, C, H, W)
        
        E_features[4] = fea_16
        
        # 样式特征提取
        gs = self.to_style(fea_16)
        
        # 如果有参考图，融合样式特征
        if self.use_reference and reference_img is not None:
            if self.lightweight_mode:
                gs = self.style_injection(gs, ref_features['global'])
            else:
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
    增加4090优化功能
    """
    def __init__(self,
                 z_dim,
                 c_dim,
                 w_dim,
                 img_resolution,
                 img_channels,
                 use_reference=True,
                 lightweight_mode=False,  # 4090优化模式
                 synthesis_kwargs={},
                 mapping_kwargs={}):
        
        # 更新synthesis_kwargs
        synthesis_kwargs.update({
            'use_reference': use_reference,
            'lightweight_mode': lightweight_mode
        })
        
        super().__init__(z_dim, c_dim, w_dim, img_resolution, img_channels,
                        synthesis_kwargs, mapping_kwargs)
        
        # 替换synthesis网络
        self.synthesis = ReferenceGuidedSynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs
        )
        
        self.lightweight_mode = lightweight_mode
        self._is_frozen = False
        
    def freeze_mat_backbone(self):
        """冻结MAT主体参数，只保留参考图模块可训练"""
        print("冻结MAT主体参数...")
        
        # 冻结mapping网络
        for param in self.mapping.parameters():
            param.requires_grad = False
            
        # 冻结synthesis网络的MAT部分
        for name, param in self.synthesis.named_parameters():
            if 'reference' not in name:  # 不冻结参考图相关模块
                param.requires_grad = False
            
        self._is_frozen = True
        
        # 显示可训练参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        return trainable_params, total_params
        
    def unfreeze_selected_layers(self, layers_to_unfreeze=None):
        """选择性解冻某些层进行微调"""
        if layers_to_unfreeze is None:
            layers_to_unfreeze = ['to_style', 'dec.final_layer']
            
        print(f"解冻选定层: {layers_to_unfreeze}")
        
        for name, param in self.named_parameters():
            for layer_name in layers_to_unfreeze:
                if layer_name in name:
                    param.requires_grad = True
                    print(f"  解冻: {name}")
                    
    def get_trainable_parameters(self):
        """获取所有可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_reference_parameters(self):
        """获取参考图模块参数"""
        params = []
        for name, param in self.named_parameters():
            if 'reference' in name and param.requires_grad:
                params.append(param)
        return params
    
    def get_backbone_parameters(self):
        """获取主干网络可训练参数"""
        params = []
        for name, param in self.named_parameters():
            if 'reference' not in name and param.requires_grad:
                params.append(param)
        return params
    
    def load_mat_weights(self, pretrained_path, strict=False):
        """加载预训练的MAT权重"""
        print(f"加载预训练MAT权重: {pretrained_path}")
        
        import legacy
        import dnnlib
        
        with dnnlib.util.open_url(pretrained_path) as f:
            pretrained_dict = legacy.load_network(f)['G_ema'].state_dict()
        
        # 只加载MAT原有的参数，忽略新增的参考图模块
        model_dict = self.state_dict()
        filtered_dict = {}
        
        for k, v in pretrained_dict.items():
            if k in model_dict and 'reference' not in k:
                filtered_dict[k] = v
                
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        print(f"成功加载 {len(filtered_dict)} 个预训练参数")
        
    def forward(self, images_in, masks_in, z, c, reference_img=None,
                truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
                noise_mode='random', return_stg1=False):
        """
        前向传播，集成参考图引导
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


def create_reference_model(pretrained_path, device='cuda', lightweight_mode=False):
    """
    创建参考图引导模型的工厂函数
    
    Args:
        pretrained_path: 预训练MAT模型路径
        device: 设备
        lightweight_mode: 是否使用4090轻量级模式
    """
    print(f"创建参考图引导模型 (轻量级模式: {lightweight_mode})...")
    
    # 创建模型
    model = ReferenceGuidedGenerator(
        z_dim=512,
        c_dim=0,
        w_dim=512, 
        img_resolution=512,
        img_channels=3,
        use_reference=True,
        lightweight_mode=lightweight_mode
    ).to(device)
    
    # 加载预训练权重
    model.load_mat_weights(pretrained_path)
    
    # 如果是轻量级模式，自动冻结主体参数
    if lightweight_mode:
        model.freeze_mat_backbone()
    
    return model
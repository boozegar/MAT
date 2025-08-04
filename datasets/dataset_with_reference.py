import cv2
import os
import numpy as np
import PIL.Image
import torch
import random
from datasets.dataset_512 import ImageFolderMaskDataset
from datasets.mask_generator_512 import RandomMask

class ReferenceGuidedDataset(ImageFolderMaskDataset):
    """
    支持参考图引导的数据集
    """
    def __init__(self,
                 path,
                 reference_path=None,     # 参考图片目录路径
                 reference_prob=0.8,      # 使用参考图的概率
                 reference_size=128,      # 参考图尺寸
                 resolution=None,
                 hole_range=[0,1],
                 **super_kwargs):
        
        super().__init__(path, resolution, hole_range, **super_kwargs)
        
        self.reference_path = reference_path
        self.reference_prob = reference_prob
        self.reference_size = reference_size
        self.use_reference = reference_path is not None
        
        # 加载参考图片列表
        if self.use_reference:
            self._load_reference_images()
            
        # mask生成器
        self.mask_generator = RandomMask(resolution, hole_range)
    
    def _load_reference_images(self):
        """加载参考图片文件名列表"""
        if os.path.isdir(self.reference_path):
            PIL.Image.init()
            self.reference_fnames = []
            for root, _dirs, files in os.walk(self.reference_path):
                for fname in files:
                    if self._file_ext(fname) in PIL.Image.EXTENSION:
                        self.reference_fnames.append(os.path.join(root, fname))
            
            if len(self.reference_fnames) == 0:
                print(f"Warning: No reference images found in {self.reference_path}")
                self.use_reference = False
        else:
            print(f"Warning: Reference path {self.reference_path} not found")
            self.use_reference = False
    
    def _load_reference_image(self, ref_path):
        """加载并预处理参考图片"""
        try:
            image = np.array(PIL.Image.open(ref_path))
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC
            
            # 转为RGB
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 4:  # RGBA -> RGB
                image = image[:, :, :3]
            
            # 调整大小到指定尺寸
            image = cv2.resize(image, (self.reference_size, self.reference_size), 
                             interpolation=cv2.INTER_AREA)
            
            # 转换为float并归一化到[-1, 1]
            image = image.astype(np.float32) / 127.5 - 1.0
            
            # HWC -> CHW
            image = image.transpose([2, 0, 1])
            
            return image
            
        except Exception as e:
            print(f"Error loading reference image {ref_path}: {e}")
            # 返回空白参考图
            return np.zeros((3, self.reference_size, self.reference_size), dtype=np.float32)
    
    def __getitem__(self, idx):
        # 获取原始图像和mask
        data = super().__getitem__(idx)  # 返回 (image, mask, label)
        image, mask = data[0], data[1]
        
        # 决定是否使用参考图
        use_ref_this_sample = (self.use_reference and 
                              random.random() < self.reference_prob)
        
        if use_ref_this_sample:
            # 随机选择一个参考图
            ref_idx = random.randint(0, len(self.reference_fnames) - 1)
            ref_path = self.reference_fnames[ref_idx]
            reference_image = self._load_reference_image(ref_path)
        else:
            # 创建空的参考图占位符
            reference_image = np.zeros((3, self.reference_size, self.reference_size), 
                                     dtype=np.float32)
        
        # 返回 (原图, mask, 参考图, 标签)
        if len(data) == 3:  # 有标签
            return image, mask, reference_image, data[2]
        else:  # 无标签
            return image, mask, reference_image


def create_reference_guided_dataset(data_path, reference_path=None, **kwargs):
    """
    创建参考图引导数据集的工厂函数
    
    Args:
        data_path: 主要训练数据路径
        reference_path: 参考图数据路径，None表示不使用参考图
        **kwargs: 其他数据集参数
    """
    return ReferenceGuidedDataset(
        path=data_path,
        reference_path=reference_path,
        **kwargs
    )
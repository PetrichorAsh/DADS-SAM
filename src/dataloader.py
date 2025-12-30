import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        # 根据实际RGB三通道值定义颜色到类别的映射
        # 类别顺序：0-Background, 1-Oil, 2-Others, 3-Water
        self.rgb_class_map = {
            (124, 0, 0): 0,    # 油污 (红色系)
            (51, 221, 255): 1, # 水域 (蓝色系)
            (255, 204, 51): 2, # 其他 (黄色)
            (0, 0, 0): 3,      # 未标注-黑
            (221, 221, 221): 3 # 未标注-灰
        }
        if mode == "train":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'images',"*.jpg"))
            self.mask_files = []
            for img_path in self.img_files:
                # 使用.png格式加载mask文件
                self.mask_files.append(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'masks', os.path.basename(img_path)[:-4] + ".png")) 

        elif mode == "test":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"],'images',"*.jpg"))
            self.mask_files = []
            for img_path in self.img_files:
                # 测试集也使用.png格式的mask文件
                self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"],'masks', os.path.basename(img_path)[:-4] + ".png"))
        else:
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["VAL_PATH"],'images',"*.jpg"))
            self.mask_files = []
            for img_path in self.img_files:
                # 测试集也使用.png格式的mask文件
                self.mask_files.append(os.path.join(config_file["DATASET"]["VAL_PATH"],'masks', os.path.basename(img_path)[:-4] + ".png"))

        self.processor = processor
        self.mode = mode
        # 创建保存可视化结果的目录
        self.vis_dir = os.path.join("visualization", mode)
        os.makedirs(self.vis_dir, exist_ok=True)

    def __len__(self):
        return len(self.img_files)
    
    def visualize_item(self, index: int):
        """可视化指定索引的图像和掩码"""
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path)
        mask = Image.open(mask_path, 'r')
        
        # 保持原始mask数据不变
        original_mask = np.array(mask, dtype=np.uint8)
        print("原始mask像素值示例:", original_mask[:3,:3])
        
        # 复制原始mask用于处理
        ground_truth_mask = original_mask.copy()
        
        # 转换到二维数组（仅处理第一个通道）
        if len(ground_truth_mask.shape) > 2:
            ground_truth_mask = ground_truth_mask[:, :, 0]
        
        # 打印原始mask唯一RGB值
        print("原始mask唯一RGB值:", np.unique(original_mask.reshape(-1, 3), axis=0))
        mapped_mask = np.zeros(original_mask.shape[:2], dtype=np.uint8)
        # 使用numpy的向量化操作进行RGB颜色映射
        for rgb, class_idx in self.rgb_class_map.items():
            rgb_array = np.array(rgb, dtype=np.uint8)
            mask_area = np.all(original_mask == rgb_array, axis=-1)
            mapped_mask[mask_area] = class_idx
        
        # 验证映射后的像素值
        unique_mapped = np.unique(mapped_mask)
        print(f"映射后mask唯一值: {unique_mapped}")

        # 设置不同类别的颜色（转换为0-1浮点数范围）
        # 类别顺序：0-Background, 1-Oil, 2-Others, 3-Water
        colors = {
            0: (1.0, 0.0, 0.486),  # 粉红色 - 油污区域 (255,0,124)
            1: (0.2, 0.866, 1.0),  # 浅蓝色 - 水域 (51,221,255)
            2: (1.0, 0.8, 0.2),     # 黄色 - 其他区域 (255,204,51)
            3: (0.0, 0.0, 0.0)     # 黑色 - 未标注区域
        }
        
        # 创建彩色掩码（浮点类型）
        colored_mask = np.zeros((*mapped_mask.shape, 3), dtype=np.float32)
        
        for class_idx, color in colors.items():
            mask_slice = (mapped_mask == class_idx)
            colored_mask[mask_slice] = color
        
        # 创建图像
        plt.figure(figsize=(12, 6), dpi=120)
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # 显示掩码（关闭归一化）
        plt.subplot(1, 2, 2)
        # 显示时指定插值方式避免像素混合
        plt.imshow(colored_mask, interpolation='nearest', vmin=0, vmax=1)
        plt.title("Segmentation Mask")
        plt.axis('off')
        
        # 保存图像
        save_path = os.path.join(self.vis_dir, f"{os.path.basename(img_path)[:-4]}_vis.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"可视化结果已保存到: {save_path}")
    
    def __getitem__(self, index: int) -> list:
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        
        # 加载图像和掩码
        image = Image.open(img_path)
        mask = Image.open(mask_path, 'r')
        
        # 调整图像和掩码尺寸到1024x1024
        target_size = (320, 320)
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
        
        # 将掩码转换为numpy数组
        ground_truth_mask = np.array(mask, dtype=np.uint8)
        
        # 统一处理为三维数组（单通道扩展为三通道）
        if len(ground_truth_mask.shape) == 2:
            ground_truth_mask = np.repeat(ground_truth_mask[:, :, np.newaxis], 3, axis=2)
        elif ground_truth_mask.shape[2] == 1:
            ground_truth_mask = np.repeat(ground_truth_mask, 3, axis=2)
        
        # 将RGB掩码映射到类别索引
        mapped_mask = np.zeros(ground_truth_mask.shape[:2], dtype=np.uint8)
        for rgb, class_idx in self.rgb_class_map.items():
            rgb_array = np.array(rgb, dtype=np.uint8)
            mask_area = np.all(ground_truth_mask == rgb_array, axis=2)
            mapped_mask[mask_area] = class_idx
        ground_truth_mask = mapped_mask
        
        # 使用调整后的尺寸作为原始尺寸
        original_size = target_size[::-1]  # (H, W)
        
        # 获取边界框
        box = utils.get_bounding_box(ground_truth_mask)
        
        # 处理输入
        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)
        return inputs
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)

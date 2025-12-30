import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
import yaml
import torch.nn.functional as F
import cv2

from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora_conv import LoRAConvSAM
from src.lora_mask_decoder import LoRA_MaskDecoder
import src.utils as utils

# 加载配置文件
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

class BoundaryIoUCalculator:
    """
    Boundary IoU计算器
    """
    def __init__(self, dilation_radius=5, kernel_size=5):
        self.dilation_radius = dilation_radius
        self.kernel_size = kernel_size
        
    def extract_boundary(self, mask):
        """提取边界"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        return boundary.astype(bool)
    
    def dilate_boundary(self, boundary):
        """膨胀边界"""
        if self.dilation_radius <= 0:
            return boundary
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2 * self.dilation_radius + 1, 2 * self.dilation_radius + 1)
        )
        dilated = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)
    
    def calculate(self, pred_mask, gt_mask):
        """计算单个类别的Boundary IoU"""
        # 提取边界
        pred_boundary = self.extract_boundary(pred_mask)
        gt_boundary = self.extract_boundary(gt_mask)
        
        # 如果没有边界，返回特殊值
        if np.sum(pred_boundary) == 0 and np.sum(gt_boundary) == 0:
            return 1.0  # 都没有边界，认为完全匹配
        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            return 0.0  # 一个有边界一个没有，完全不匹配
        
        # 膨胀真实边界，用于容忍预测边界的小偏差
        gt_boundary_dilated = self.dilate_boundary(gt_boundary)
        
        # 计算预测边界与膨胀后真实边界的重叠
        # 这种方法更符合边界质量评估的直觉
        intersection = np.logical_and(pred_boundary, gt_boundary_dilated)
        
        # 计算召回率：预测边界中有多少在真实边界的容忍范围内
        pred_boundary_dilated = self.dilate_boundary(pred_boundary)
        recall_intersection = np.logical_and(gt_boundary, pred_boundary_dilated)
        
        # 使用F1-score风格的计算
        precision = np.sum(intersection) / np.sum(pred_boundary) if np.sum(pred_boundary) > 0 else 0.0
        recall = np.sum(recall_intersection) / np.sum(gt_boundary) if np.sum(gt_boundary) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        # 返回F1-score作为Boundary IoU
        boundary_iou = 2 * precision * recall / (precision + recall)
        return boundary_iou

def calculate_multiclass_boundary_iou(pred_masks, gt_masks, num_classes=4, dilation_radius=5):
    """
    计算多类别的Boundary IoU
    
    Args:
        pred_masks: 预测掩码 (H, W) - 类别索引
        gt_masks: 真实掩码 (H, W) - 类别索引
        num_classes: 类别数量
        dilation_radius: 边界膨胀半径
    
    Returns:
        class_boundary_ious: 每个类别的Boundary IoU列表
        mean_boundary_iou: 平均Boundary IoU
    """
    calculator = BoundaryIoUCalculator(dilation_radius=dilation_radius)
    class_boundary_ious = []
    
    for cls in range(num_classes):
        # 为每个类别创建二值掩码
        pred_binary = (pred_masks == cls)
        gt_binary = (gt_masks == cls)
        
        # 计算该类别的Boundary IoU
        boundary_iou = calculator.calculate(pred_binary, gt_binary)
        class_boundary_ious.append(boundary_iou)
    
    # 计算平均Boundary IoU
    mean_boundary_iou = np.mean(class_boundary_ious)
    
    return class_boundary_ious, mean_boundary_iou

# 加载SAM模型
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])

# 创建图像编码器的LoRA适配器
sam_lora_conv = LoRAConvSAM(sam, config_file["SAM"]["RANK"])  
model = sam_lora_conv.sam

# 创建mask decoder的LoRA适配器
mask_decoder_lora = LoRA_MaskDecoder(
    mask_decoder=model.mask_decoder,
    rank=config_file["SAM"]["RANK"],
    num_classes=4  # 四分类任务
)
model.mask_decoder = mask_decoder_lora

# 处理数据集
processor = Samprocessor(model)
test_ds = DatasetSegmentation(config_file, processor, mode="test")

# 创建数据加载器
test_dataloader = DataLoader(
    test_ds,
    batch_size=config_file["TRAIN"]["BATCH_SIZE"],
    shuffle=False,
    collate_fn=collate_fn
)

def calculate_metrics(pred_mask, gt_mask):
    """
    计算各类别的指标：IoU, Precision, Recall, F1 Score, Boundary IoU
    """
    # 确保张量在同一设备上
    device = pred_mask.device
    pred_mask = pred_mask.to(device)
    gt_mask = gt_mask.to(device)
    
    # 将预测结果转换为类别索引
    pred_mask = torch.argmax(pred_mask, dim=1)  # [B, H, W]
    gt_mask = gt_mask.squeeze(1)  # [B, H, W]
    
    batch_size = pred_mask.shape[0]
    num_classes = 4
    
    metrics = {
        'iou': [[] for _ in range(num_classes)],
        'precision': [[] for _ in range(num_classes)],
        'recall': [[] for _ in range(num_classes)],
        'f1': [[] for _ in range(num_classes)],
        'accuracy': [[] for _ in range(num_classes)],
        'boundary_iou': [[] for _ in range(num_classes)],  # 添加Boundary IoU
        'pixel_accuracy': []  # 整体像素级准确率
    }
    
    for b in range(batch_size):
        # 计算Boundary IoU（每个样本一次）
        pred_mask_cpu = pred_mask[b].cpu().numpy()
        gt_mask_cpu = gt_mask[b].cpu().numpy()
        
        # 计算每个类别的Boundary IoU
        class_boundary_ious, _ = calculate_multiclass_boundary_iou(
            pred_mask_cpu, gt_mask_cpu, num_classes=4, dilation_radius=5
        )
        
        for cls in range(num_classes):
            pred_cls = (pred_mask[b] == cls)
            gt_cls = (gt_mask[b] == cls)
            
            intersection = torch.logical_and(pred_cls, gt_cls).sum().float()
            union = torch.logical_or(pred_cls, gt_cls).sum().float()
            pred_sum = pred_cls.sum().float()
            gt_sum = gt_cls.sum().float()
            
            # IoU
            iou = (intersection + 1e-6) / (union + 1e-6)
            metrics['iou'][cls].append(iou.item())
            
            # Precision
            precision = (intersection + 1e-6) / (pred_sum + 1e-6)
            metrics['precision'][cls].append(precision.item())
            
            # Recall
            recall = (intersection + 1e-6) / (gt_sum + 1e-6)
            metrics['recall'][cls].append(recall.item())
            
            # F1 Score
            f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)
            metrics['f1'][cls].append(f1.item())
            
            # Accuracy (标准准确率：正确预测的像素数除以总像素数)
            # 对于每个类别，我们计算该类别预测正确的像素比例
            # 使用全图像范围计算准确率
            correct_pixels = torch.eq(pred_cls, gt_cls).sum().float()
            total_pixels = pred_cls.numel()
            accuracy = correct_pixels / total_pixels
            metrics['accuracy'][cls].append(accuracy.item())
            
            # Boundary IoU
            metrics['boundary_iou'][cls].append(class_boundary_ious[cls])
    
    # 计算整体像素级准确率
    for b in range(batch_size):
        correct_pixels = (pred_mask[b] == gt_mask[b]).sum().float()
        total_pixels = pred_mask[b].numel()
        pixel_accuracy = correct_pixels / total_pixels
        metrics['pixel_accuracy'].append(pixel_accuracy.item())
    
    return metrics

def calculate_mean_metrics(metrics):
    """
    计算各类别的平均指标
    """
    num_classes = 4
    mean_metrics = {}
    
    for metric_name in ['iou', 'precision', 'recall', 'f1', 'accuracy', 'boundary_iou']:
        mean_metrics[metric_name] = []
        for cls in range(num_classes):
            if metrics[metric_name][cls]:
                mean_val = sum(metrics[metric_name][cls]) / len(metrics[metric_name][cls])
                mean_metrics[metric_name].append(mean_val)
            else:
                mean_metrics[metric_name].append(0.0)
    
    # 处理pixel_accuracy
    if metrics['pixel_accuracy']:
        mean_metrics['pixel_accuracy'] = sum(metrics['pixel_accuracy']) / len(metrics['pixel_accuracy'])
    else:
        mean_metrics['pixel_accuracy'] = 0.0
    
    return mean_metrics

def print_metrics(mean_metrics):
    """
    打印各类别和平均指标
    """
    class_names = ['Oil', 'Water', 'Others', 'Background']
    num_classes = 4
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS BY CLASS")
    print("="*80)
    
    for cls in range(num_classes):
        print(f"\n{class_names[cls]} (Class {cls}):")
        print(f"  IoU:         {mean_metrics['iou'][cls]:.4f}")
        print(f"  Boundary IoU:{mean_metrics['boundary_iou'][cls]:.4f}")
        print(f"  Precision:   {mean_metrics['precision'][cls]:.4f}")
        print(f"  Recall:      {mean_metrics['recall'][cls]:.4f}")
        print(f"  F1-Score:    {mean_metrics['f1'][cls]:.4f}")
        print(f"  Accuracy:    {mean_metrics['accuracy'][cls]:.4f}")
    
    # 计算平均指标（忽略背景类）
    print("\n" + "-"*80)
    print("AVERAGE METRICS (excluding background)")
    print("-"*80)
    
    avg_iou = sum(mean_metrics['iou'][:3]) / 3
    avg_boundary_iou = sum(mean_metrics['boundary_iou'][:3]) / 3
    avg_precision = sum(mean_metrics['precision'][:3]) / 3
    avg_recall = sum(mean_metrics['recall'][:3]) / 3
    avg_f1 = sum(mean_metrics['f1'][:3]) / 3
    avg_accuracy = sum(mean_metrics['accuracy'][:3]) / 3
    
    print(f"Average IoU:         {avg_iou:.4f}")
    print(f"Average Boundary IoU:{avg_boundary_iou:.4f}")
    print(f"Average Precision:   {avg_precision:.4f}")
    print(f"Average Recall:      {avg_recall:.4f}")
    print(f"Average F1-Score:    {avg_f1:.4f}")
    print(f"Average Accuracy:    {avg_accuracy:.4f}")
    
    # 总体平均指标（包含背景类）
    print("\n" + "-"*80)
    print("OVERALL AVERAGE METRICS (including background)")
    print("-"*80)
    
    overall_iou = sum(mean_metrics['iou']) / 4
    overall_boundary_iou = sum(mean_metrics['boundary_iou']) / 4
    overall_precision = sum(mean_metrics['precision']) / 4
    overall_recall = sum(mean_metrics['recall']) / 4
    overall_f1 = sum(mean_metrics['f1']) / 4
    overall_accuracy = sum(mean_metrics['accuracy']) / 4
    
    print(f"Overall IoU:         {overall_iou:.4f}")
    print(f"Overall Boundary IoU:{overall_boundary_iou:.4f}")
    print(f"Overall Precision:   {overall_precision:.4f}")
    print(f"Overall Recall:      {overall_recall:.4f}")
    print(f"Overall F1-Score:    {overall_f1:.4f}")
    print(f"Overall Accuracy:    {overall_accuracy:.4f}")
    
    # 整体像素级准确率
    print(f"\nPixel-level Accuracy: {mean_metrics['pixel_accuracy']:.4f}")
    print("="*80)

def load_model_weights(model_path_decoder, model_path_lora):
    """
    加载模型权重
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载LoRA权重到图像编码器
    sam_lora_conv.load_parameters(model_path_lora)
    
    # 加载LoRA权重到mask decoder
    mask_decoder_lora.load_lora_parameters(model_path_decoder)
    
    # 确保所有参数在正确的设备上
    model.to(device)
    
    print(f"Loaded model weights from:")
    print(f"  Decoder: {model_path_decoder}")
    print(f"  LoRA:    {model_path_lora}")

def run_inference(model_path_decoder, model_path_lora):
    """
    运行推理并计算指标
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # 加载模型权重
    load_model_weights(model_path_decoder, model_path_lora)
    
    # 初始化指标存储
    all_metrics = {
        'iou': [[] for _ in range(4)],
        'precision': [[] for _ in range(4)],
        'recall': [[] for _ in range(4)],
        'f1': [[] for _ in range(4)],
        'accuracy': [[] for _ in range(4)],
        'boundary_iou': [[] for _ in range(4)],
        'pixel_accuracy': []
    }
    
    print("Running inference on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            # 确保批次数据在正确的设备上
            for i in range(len(batch)):
                for key in batch[i]:
                    if isinstance(batch[i][key], torch.Tensor):
                        batch[i][key] = batch[i][key].to(device)
            
            outputs = model(
                batched_input=batch,
                multimask_output=False
            )
            
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            # 确保所有张量都在同一设备上
            stk_out = stk_out.squeeze(1).to(device)  # [B, num_classes, H, W]
            stk_gt = stk_gt.unsqueeze(1).long().to(device)  # [B, 1, H, W] with class indices
            
            # 计算指标
            batch_metrics = calculate_metrics(stk_out, stk_gt)
            
            # 累积指标
            for metric_name in ['iou', 'precision', 'recall', 'f1', 'accuracy', 'boundary_iou']:
                for cls in range(4):
                    all_metrics[metric_name][cls].extend(batch_metrics[metric_name][cls])
            
            # 累积pixel_accuracy
            all_metrics['pixel_accuracy'].extend(batch_metrics['pixel_accuracy'])
    
    # 计算平均指标
    mean_metrics = calculate_mean_metrics(all_metrics)
    
    # 打印结果
    print_metrics(mean_metrics)
    
    return mean_metrics

if __name__ == "__main__":
    # 指定模型权重路径
    model_path_decoder = ""
    model_path_lora = ""
    
    # 运行推理
    mean_metrics = run_inference(model_path_decoder, model_path_lora)
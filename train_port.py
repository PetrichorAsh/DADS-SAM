import torch
import torch.nn as nn
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
from src.lora_conv import LoRAConvSAM
from src.lora_mask_decoder import LoRA_MaskDecoder
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

# Focal Loss定义
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=4, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # inputs: [B, num_classes, H, W]
        # targets: [B, 1, H, W] with class indices
        
        # 将targets转换为one-hot编码
        targets = targets.squeeze(1)  # [B, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # [B, num_classes, H, W]
        
        # 计算softmax概率
        p = F.softmax(inputs, dim=1)
        
        # 计算focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B, H, W]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 加载配置文件
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# 加载SAM模型
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])

# 先加载预训练权重
sam.load_state_dict(torch.load(config_file["SAM"]["CHECKPOINT"]))

# 创建图像编码器的LoRA适配器
sam_lora_conv = LoRAConvSAM(sam, config_file["SAM"]["RANK"])  
model = sam_lora_conv.sam

# model = sam

# 创建mask decoder的LoRA适配器
mask_decoder_lora = LoRA_MaskDecoder(
    mask_decoder=model.mask_decoder,
    rank=config_file["SAM"]["RANK"],
    num_classes=4  # 四分类任务
)
model.mask_decoder = mask_decoder_lora
# print(model)

# 处理数据集
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")
val_ds = DatasetSegmentation(config_file, processor, mode="val")

# 创建数据加载器
train_dataloader = DataLoader(
    train_ds,
    batch_size=config_file["TRAIN"]["BATCH_SIZE"],
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_ds,
    batch_size=config_file["TRAIN"]["BATCH_SIZE"],
    shuffle=False,
    collate_fn=collate_fn
)

def calculate_iou(pred_mask, gt_mask):
    # 将预测结果转换为类别索引
    pred_mask = torch.argmax(pred_mask, dim=1)  # [B, H, W]
    gt_mask = gt_mask.squeeze(1)  # [B, H, W]
    
    total_iou = 0
    batch_size = pred_mask.shape[0]
    num_classes = 4
    class_ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = torch.logical_and(pred_cls, gt_cls).sum()
        union = torch.logical_or(pred_cls, gt_cls).sum()
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        class_ious.append(iou.item())
        total_iou += iou.item()
    
    # 返回每个类别的IoU和平均IoU
    return class_ious, total_iou / num_classes

# 初始化优化器
optimizer = Adam([
    {"params": model.image_encoder.parameters(), "lr": config_file["TRAIN"]["LEARNING_RATE"]},
    {"params": mask_decoder_lora.parameters(), "lr": config_file["TRAIN"]["LEARNING_RATE"]}
], weight_decay=1e-4)

# # 使用CrossEntropyLoss作为多分类损失函数
# seg_loss = nn.CrossEntropyLoss()
# 使用DiceLoss和FocalLoss的组合作为多分类损失函数
dice_loss = monai.losses.DiceCELoss(softmax=True, to_onehot_y=True, squared_pred=True, reduction='mean')
focal_loss = FocalLoss(alpha=1, gamma=2, num_classes=4, reduction='mean')

# 损失函数权重
dice_weight = 0.7
focal_weight = 0.3
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.train()
model.to(device)

total_loss = []
# 初始化最佳IoU值
best_iou = 0.0
rank = config_file["SAM"]["RANK"]

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):

        outputs = model(
            batched_input=batch,
            multimask_output=False
        )

        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)  # [B, num_classes, H, W]
        stk_gt = stk_gt.unsqueeze(1).long().to(device)  # [B, 1, H, W] with class indices

        # 计算Dice Loss和Focal Loss
        d_loss = dice_loss(stk_out, stk_gt)
        f_loss = focal_loss(stk_out, stk_gt)
        
        # 组合损失
        loss = dice_weight * d_loss + focal_weight * f_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')
    
    # 验证阶段
    model.eval()
    val_ious = []
    val_class_ious = [[] for _ in range(4)]  # 存储每个类别的IoU
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            outputs = model(
                batched_input=batch,
                multimask_output=False
            )
            
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)  # [B, num_classes, H, W]
            stk_gt = stk_gt.unsqueeze(1).long().to(device)  # [B, 1, H, W]
            
            class_ious, batch_iou = calculate_iou(stk_out, stk_gt)
            val_ious.append(batch_iou)
            
            # 存储每个类别的IoU
            for cls in range(4):
                val_class_ious[cls].append(class_ious[cls])
    
    # 计算每个类别的平均IoU和总体miou
    mean_class_ious = [mean(cls_ious) for cls_ious in val_class_ious]
    mean_iou = mean(val_ious)
    
    # 获取oil iou（类别0的IoU）
    oil_iou = mean_class_ious[0]
    miou = mean_iou
    
    print(f'Validation Mean IoU: {mean_iou:.4f}')
    print(f'Oil IoU: {oil_iou:.4f}')

    # 保存当前epoch的模型参数，文件名包含oil iou和miou值
    oil_iou_str = f"{oil_iou:.4f}".replace(".", "p")  # 将小数点替换为p以避免文件名问题
    miou_str = f"{miou:.4f}".replace(".", "p")
    
    print(f'保存模型权重，Oil IoU: {oil_iou:.4f}, mIoU: {miou:.4f}')
    sam_lora_conv.save_parameters(f"loraconv_epoch{epoch}_oil_iou{oil_iou_str}_miou{miou_str}.pth")
    mask_decoder_lora.save_lora_parameters(f"decoder_epoch{epoch}_oil_iou{oil_iou_str}_miou{miou_str}_decoder.safetensors")


# 保存最后一轮的模型参数
sam_lora_conv.save_parameters(f"loraconv_final_lora_rank{rank}_epoch{num_epochs-1}.pth")
mask_decoder_lora.save_lora_parameters(f"decoder_final_lora_rank{rank}_epoch{num_epochs-1}_decoder.safetensors")
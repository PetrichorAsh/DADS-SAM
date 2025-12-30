import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from safetensors import safe_open
from safetensors.torch import save_file
from src.segment_anything.modeling.mask_decoder import MaskDecoder

class LoRA_MaskDecoder(MaskDecoder):
    """
    为mask decoder添加LoRA适配器以支持四分类任务
    
    Arguments:
        mask_decoder: 原始的mask decoder模块
        rank: LoRA的秩
        num_classes: 分类数量(默认为4)
    """
    def __init__(self, mask_decoder: MaskDecoder, rank: int, num_classes: int = 4):
        # 保持原始MaskDecoder参数不变
        super().__init__(
            transformer_dim=mask_decoder.transformer_dim,
            transformer=mask_decoder.transformer,
            num_multimask_outputs=mask_decoder.num_multimask_outputs,  # 保持原始值
            activation=type(mask_decoder.output_upscaling[2]),
            iou_head_depth=3,  # 恢复默认值
            iou_head_hidden_dim=256  # 恢复默认值
        )
        
        # 只加载兼容的预训练参数
        pretrained_dict = mask_decoder.state_dict()
        model_dict = self.state_dict()
        # 过滤不匹配的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        # 添加LoRA适配器
        self.rank = rank
        self.num_classes = num_classes
        transformer_dim = self.transformer_dim
        self.lora_A = nn.Linear(transformer_dim // 8, rank, bias=False)
        self.lora_B = nn.Linear(rank, num_classes, bias=False)  # 四分类输出通道
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
        # 冻结原始参数
        for name, param in self.named_parameters():
            if not any(x in name for x in ["lora_A", "lora_B"]):
                param.requires_grad = False
        
        # 仅训练LoRA参数
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取原始MaskDecoder的输出
        _, iou_pred = super().forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
        
        # 获取特征用于LoRA分支
        tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens = tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((tokens, sparse_prompt_embeddings), dim=1)
        
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        
        hs, src = self.transformer(src, pos_src, tokens)
        src = src.transpose(1, 2).view(src.shape[0], -1, *image_embeddings.shape[-2:])
        
        # 获取特征用于LoRA分支，并调整维度
        upscaled_embedding = self.output_upscaling[:-1](src)
        b, c, h, w = upscaled_embedding.shape
        
        # 调整特征维度以适应LoRA处理
        features = upscaled_embedding.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        features = features.reshape(-1, c)  # [B*H*W, C//8] 调整输入维度
        
        # 应用LoRA变换
        lora_features = self.lora_A(features)  # [B*H*W, rank]
        lora_output = self.lora_B(lora_features)  # [B*H*W, num_classes]
        
        # 重塑为四分类输出 (batch, num_classes, H, W)
        lora_output = lora_output.view(b, h * w, self.num_classes).permute(0, 2, 1)  # [B, num_classes, H*W]
        lora_output = lora_output.view(b, self.num_classes, h, w)  # [B, num_classes, H, W]
        
        # 添加上采样操作，将256x256调整到1024x1024
        lora_output = F.interpolate(lora_output, size=(320, 320), mode='bilinear', align_corners=False)
        
        # 移除多余的维度，确保输出维度为[B, num_classes, H, W]
        if lora_output.dim() > 4:
            lora_output = lora_output.squeeze(1)
        # print(f"LoRA输出维度: {lora_output.shape}")
        return lora_output, iou_pred
    
    def save_lora_parameters(self, filename: str):
        """保存LoRA参数"""
        lora_state_dict = {
            "lora_A.weight": self.lora_A.weight,
            "lora_B.weight": self.lora_B.weight
        }
        save_file(lora_state_dict, filename)
    
    def load_lora_parameters(self, filename: str):
        """加载LoRA参数"""
        device = next(self.parameters()).device  # 获取当前设备
        with safe_open(filename, framework="pt") as f:
            lora_a_weight = f.get_tensor("lora_A.weight").to(device)
            lora_b_weight = f.get_tensor("lora_B.weight").to(device)
            self.lora_A.weight = nn.Parameter(lora_a_weight)
            self.lora_B.weight = nn.Parameter(lora_b_weight)

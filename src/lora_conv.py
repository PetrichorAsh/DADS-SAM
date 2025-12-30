from src.segment_anything import build_sam_vit_b
from src.segment_anything.modeling.sam import Sam

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
import yaml

class LightConvAdapter(nn.Module):
    """
    轻量高效的ConvAdapter - 使用深度可分离卷积和动态特征聚合
    """
    def __init__(self, dim, reduction_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim // reduction_factor
        
        # 轻量化的降维投影
        self.down_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 深度可分离卷积模块
        self.depth_conv = nn.ModuleDict({
            'small': nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            'large': nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim)
        })
        
        # 轻量化的点卷积
        self.point_conv = nn.ModuleDict({
            'small': nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            'large': nn.Conv1d(hidden_dim, hidden_dim//2, 1)
        })
        
        # 动态特征聚合
        self.dynamic_mixer = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim//4, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim//4, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征重校准
        self.recalibration = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 上采样投影
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # 自适应残差连接
        self.res_gate = nn.Parameter(torch.zeros(1))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        def _init_conv(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            else:
                _init_conv(m)
    
    def forward(self, x):
        identity = x
        
        # 降维
        h = self.down_proj(x)  # [B, N, C]
        
        if len(h.shape) == 3:
            B, N, C = h.shape
            h = h.transpose(1, 2)  # [B, C, N]
            
            # 多尺度特征提取
            feat_s = self.point_conv['small'](
                self.depth_conv['small'](h)
            )
            feat_l = self.point_conv['large'](
                self.depth_conv['large'](h)
            )
            
            # 特征融合
            feat = torch.cat([feat_s, feat_l], dim=1)  # [B, C, N]
            
            # 通道注意力
            channel_weights = self.channel_attention(feat)
            feat = feat * channel_weights
            
            # 转回原始维度
            h = feat.transpose(1, 2)  # [B, N, C]
            
            # 动态特征聚合权重
            mix_weights = self.dynamic_mixer(h.mean(dim=1))  # [B, 2]
            h = mix_weights[:, 0:1, None] * feat_s.transpose(1, 2) + \
                mix_weights[:, 1:2, None] * feat_l.transpose(1, 2)
            
            # 特征重校准
            h = self.recalibration(h)
        
        # 上采样
        output = self.up_proj(h)
        
        # 自适应残差连接
        gate = torch.sigmoid(self.res_gate) * 0.5
        output = gate * output + (1 - gate) * identity
        
        return output

class LoRA_qkv(nn.Module):
    """
    LoRA adaption for attention modules
    """
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features

    def forward(self, x: Tensor):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba
        qkv[:, :, :, -self.d_model:] += v_ba
        return qkv

class LoRAConvSAM(nn.Module):
    """
    整合LoRA和ConvAdapter的SAM模型
    """
    def __init__(
        self,
        sam_model: Sam,
        rank: int,
        conv_reduction_factor: int = 4,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        adapter_layers = None
    ):
        super().__init__()
        self.rank = rank
        self.sam = sam_model
        
        # 初始化LoRA
        self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        self.A_weights = []
        self.B_weights = []
        
        # 冻结图像编码器参数
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        
        # 添加LoRA到注意力模块
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
                
            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features
            
            # 创建LoRA权重
            w_a_linear_q = nn.Linear(self.d_model, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, self.d_model, bias=False)
            w_a_linear_v = nn.Linear(self.d_model, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, self.d_model, bias=False)
            
            self.A_weights.extend([w_a_linear_q, w_a_linear_v])
            self.B_weights.extend([w_b_linear_q, w_b_linear_v])
            
            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear, w_a_linear_q, w_b_linear_q,
                w_a_linear_v, w_b_linear_v
            )
        
        # 初始化LoRA权重
        self.reset_lora_parameters()
        
        # 添加ConvAdapter
        if adapter_layers is None:
            adapter_layers = [4, 7, 10]  # 默认在中间层添加adapter
        
        self.adapter_layers = adapter_layers
        for i in adapter_layers:
            block = sam_model.image_encoder.blocks[i]
            block.adapter = LightConvAdapter(
                dim=self.d_model,
                reduction_factor=conv_reduction_factor,
                dropout=dropout
            )
            # print(block.adapter)
    
    def reset_lora_parameters(self):
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)
    
    def save_parameters(self, save_path):
        """保存LoRA和ConvAdapter参数"""
        state_dict = {
            'lora': {
                'A_weights': [w.state_dict() for w in self.A_weights],
                'B_weights': [w.state_dict() for w in self.B_weights]
            },
            'adapter': {
                f'adapter_{i}': self.sam.image_encoder.blocks[i].adapter.state_dict()
                for i in self.adapter_layers
            }
        }
        torch.save(state_dict, save_path)
    
    def load_parameters(self, load_path):
        """加载LoRA和ConvAdapter参数"""
        device = next(self.sam.parameters()).device  # 获取当前设备
        state_dict = torch.load(load_path, map_location=device)
        
        # 加载LoRA参数
        for w_A, w_state in zip(self.A_weights, state_dict['lora']['A_weights']):
            w_A.load_state_dict(w_state)
        for w_B, w_state in zip(self.B_weights, state_dict['lora']['B_weights']):
            w_B.load_state_dict(w_state)
            
        # 加载Adapter参数
        for i in self.adapter_layers:
            self.sam.image_encoder.blocks[i].adapter.load_state_dict(
                state_dict['adapter'][f'adapter_{i}']
            )
    
    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)

with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

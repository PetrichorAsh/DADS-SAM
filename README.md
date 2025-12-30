# DADS-SAM 溢油检测系统

基于 Segment Anything Model (SAM) 的溢油检测深度学习项目，使用 DALoRA (dynamic adaptive low-rank adaptation) 和 DSCA (dual scale convolutional adapter) 进行参数高效微调。

## 项目简介

本项目实现了基于 SAM 模型的溢油检测系统，能够对图像中的油污、水域、其他区域和背景进行精确分割。项目采用 PEFT 技术对预训练的 SAM-ViT-B 模型进行微调，大大减少了训练参数量，提高了训练效率。

## 项目结构

```
├── src/
│   ├── dataloader.py          # 数据加载和预处理
│   ├── processor.py           # SAM 输入处理器
│   ├── lora.py               # LoRA 适配器基础类
│   ├── lora_conv.py          # 图像编码器 DALoRA DSCA 适配器
│   ├── lora_mask_decoder.py  # Mask解码器 MCSH 适配器
│   ├── utils.py              # 工具函数
│   └── segment_anything/     # SAM 官方源码
├── train_port.py            # 训练脚本
├── inference_metrics.py     # 推理和评估脚本
├── config.yaml              # 配置文件
└── README.md                # 项目说明
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.9.0

## 主要依赖

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
monai>=1.0.0
gradio>=3.0.0
matplotlib>=3.0.0
PyYAML>=5.0.0
tqdm>=4.0.0
```

## 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd DADS-SAM
   ```

2. **安装依赖**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install monai opencv-python Pillow gradio matplotlib PyYAML tqdm
   ```

3. **下载 SAM 预训练权重**
   - 下载 `sam_vit_b_01ec64.pth` 并放置在项目根目录
   - 更新 `config.yaml` 中的 `SAM.CHECKPOINT` 路径

## 配置说明

编辑 `config.yaml` 文件：

```yaml
DATASET:
  TRAIN_PATH: "path/to/train/data"
  TEST_PATH: "path/to/test/data"
  VAL_PATH: "path/to/val/data"

SAM:
  CHECKPOINT: "./sam_vit_b_01ec64.pth"
  RANK: 512  # LoRA 秩参数

TRAIN:
  BATCH_SIZE: 2
  NUM_EPOCHS: 100
  LEARNING_RATE: 0.0001
```

## 使用方法

### 1. 训练模型

```bash
python train_port.py
```

### 2. 评估模型

```python
from inference_metrics import run_inference

# 指定模型权重路径
model_path_decoder = "path/to/mask_decoder_weights.pth"
model_path_lora = "path/to/lora_weights.safetensors"

# 运行推理并计算指标
mean_metrics = run_inference(model_path_decoder, model_path_lora)
```

## 引用

如果您在研究中使用了本项目，请考虑引用：

```bibtex
@article{sam2023,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

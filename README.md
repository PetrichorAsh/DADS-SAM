# DADS-SAM: DALoRA and DSCA Enhanced Fine-tuning for Port Oil Spill Detection Using UAV Imagery

A deep learning project for oil spill detection based on Segment Anything Model (SAM), using DALoRA (dynamic adaptive low-rank adaptation) and DSCA (dual scale convolutional adapter) for parameter-efficient fine-tuning.

## Project Introduction

This project implements an oil spill detection system based on the SAM model, capable of precisely segmenting oil spills, water areas, other regions, and backgrounds in images. The project uses PEFT technology to fine-tune the pre-trained SAM-ViT-B model, significantly reducing the number of training parameters and improving training efficiency.

## Project Structure

```
├── src/
│   ├── dataloader.py          # Data loading and preprocessing
│   ├── processor.py           # SAM input processor
│   ├── lora.py               # LoRA adapter base class
│   ├── lora_conv.py          # Image encoder DALoRA DSCA adapter
│   ├── lora_mask_decoder.py  # Mask decoder MCSH adapter
│   ├── utils.py              # Utility functions
│   └── segment_anything/     # Official SAM source code
├── train_port.py            # Training script
├── inference_metrics.py     # Inference and evaluation script
├── config.yaml              # Configuration file
└── README.md                # Project documentation
```

## Environment Requirements

- Python >= 3.8
- PyTorch >= 1.9.0

## Main Dependencies

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

## Installation Steps

1. **Clone the project**
   ```bash
   git clone <repository-url>
   cd DADS-SAM
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install monai opencv-python Pillow gradio matplotlib PyYAML tqdm
   ```

3. **Download SAM pre-trained weights**
   - Download `sam_vit_b_01ec64.pth` and place it in the project root directory
   - Update the `SAM.CHECKPOINT` path in `config.yaml`

## Configuration Instructions

Edit the `config.yaml` file:

```yaml
DATASET:
  TRAIN_PATH: "path/to/train/data"
  TEST_PATH: "path/to/test/data"
  VAL_PATH: "path/to/val/data"

SAM:
  CHECKPOINT: "./sam_vit_b_01ec64.pth"
  RANK: 512  # LoRA rank parameter

TRAIN:
  BATCH_SIZE: 2
  NUM_EPOCHS: 100
  LEARNING_RATE: 0.0001
```

## Usage Instructions

### 1. Train the model

```bash
python train_port.py
```

### 2. Evaluate the model

```python
from inference_metrics import run_inference

# Specify model weight paths
model_path_decoder = "path/to/mask_decoder_weights.pth"
model_path_lora = "path/to/lora_weights.safetensors"

# Run inference and calculate metrics
mean_metrics = run_inference(model_path_decoder, model_path_lora)
```

## Citation

If you use this project in your research, please consider citing:

```bibtex
@ARTICLE{11318053,
  author={Guo, Shen and Li, Ying and Shang, Jiashuo and Wang, Zi and Yuan, Jingyi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DADS-SAM: DALoRA and DSCA Enhanced Fine-tuning for Port Oil Spill Detection Using UAV Imagery}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Parameter-efficient fine-tuning;oil spill detection;segment anything model;remote sensing},
  doi={10.1109/TGRS.2025.3649271}}
```

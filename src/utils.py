import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.functional import pad


def show_mask(mask: np.array, ax, random_color=False):
    """
    Plot the mask with different colors for different classes

    Arguments:
        mask: Array of the mask with class indices (0-3)
    """
    colors = {
        0: np.array([255/255, 0/255, 124/255, 0.6]),    # 粉红色 - 油污区域
        1: np.array([51/255, 221/255, 255/255, 0.6]),    # 浅蓝色 - 水域
        2: np.array([255/255, 204/255, 51/255, 0.6]),    # 黄色 - 其他区域
        3: np.array([0/255, 0/255, 0/255, 0.6])         # 黑色 - 未标注区域
    }
    
    h, w = mask.shape[:2]
    mask_image = np.zeros((h, w, 4), dtype=np.float32)
    
    for class_idx, color in colors.items():
        class_mask = (mask == class_idx)
        mask_image[class_mask] = color
    
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    """
    Plot the image and the mask superposed

    Arguments:
        image: PIL original image
        mask: PIL original binary mask
    """
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    if len(ground_truth_seg.shape) > 2:
        ground_truth_seg = ground_truth_seg[:, :, 0]
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("./plots/" + filename + ".jpg")
    plt.close()
    

def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    """
    Take an image from the dataset and plot it

    Arguments:
        dataset: Dataset class loaded with our images
        idx: Index of the data we want
    """
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    plot_image_mask(image, mask)


def get_bounding_box(ground_truth_map: np.array) -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
  # get bounding box from mask, considering all foreground classes (class = 0, 1, 2)
  idx = np.where((ground_truth_map == 0) | (ground_truth_map == 1) | (ground_truth_map == 2))
  x_indices = idx[1]
  y_indices = idx[0]
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


def stacking_batch(batch, outputs):
    """
    Given the batch and outputs of SAM, stacks the tensors to compute the loss. We stack by adding another dimension.

    Arguments:
        batch(list(dict)): List of dict with the keys given in the dataset file
        outputs: list(dict): List of dict that are the outputs of SAM
    
    Return: 
        stk_gt: Stacked tensor of the ground truth masks in the batch. Shape: [batch_size, H, W] -> We will need to add the channels dimension (dim=1)
        stk_out: Stacked tensor of logits mask outputed by SAM. Shape: [batch_size, 1, 1, H, W] -> We will need to remove the extra dimension (dim=1) needed by SAM 
    """
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        
    return stk_gt, stk_out
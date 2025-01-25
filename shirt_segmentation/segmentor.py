import os
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from colour import Color
import warnings
from skimage import color
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.datapoints")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.v2")
from shirt_segmentation.segmentation_utils import *




# ref: https://huggingface.co/mattmdjaga/segformer_b2_clothes
def segmentor(img, processor, model, invert, device):
    
    # Process image with SegFormer
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    
    # Upsample logits to match original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    # Get segmentation prediction and create shirt mask
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    shirt_unit_function = (pred_seg == 4).cpu().numpy() # 4 is for upper-clothes


    # convert PIL image to numpy array for post processing
    img_array = np.array(img)

    # Convert to greyscale
    greyscale = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

    greyscale_masked = shirt_unit_function * greyscale

    if invert:
        # standard procedure: post processing
        shirt_mask = post_process(greyscale_masked, shirt_unit_function)
    else:
        # Resort to no post processing because it would mean that greyscale_masked has already very few or no non-zero values
        shirt_mask = greyscale_masked

    rgb_result=get_color(img_array,shirt_mask)
    mean_r, mean_g, mean_b = rgb_result[0], rgb_result[1], rgb_result[2]


    color_name = get_closest_color_name(mean_r, mean_g, mean_b)

    return color_name, (mean_r, mean_g, mean_b)


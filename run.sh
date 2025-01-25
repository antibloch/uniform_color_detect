#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropiate environment


conda activate color_detect


python main.py  --video_pth $1  --output_pth output_imgs --percent_frames 1.0 --fps 24
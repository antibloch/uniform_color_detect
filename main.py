import torchvision
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.deprecation")
from ultralytics import YOLO
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from rich.progress import Progress, BarColumn, TextColumn
import torch
import cv2
import os
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from person_detection.detector import *
from shirt_segmentation.segmentor import *
from utils.main_utils import *


def main(video_pth, output_pth, percent_frames, human_detector, cloth_processor, cloth_segmenter, fps, device):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    else:
        # remove all files instead out_dir
        for file in os.listdir(output_pth):
            os.remove(os.path.join(output_pth, file))

    # Open the video file
    cap = cv2.VideoCapture(video_pth)
    if not cap.isOpened():
        print("Error opening the  video file. Either path is correct or video is corrupted")
        return

    # Get total number of frames
    K = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {K}")

    # Compute the number of frames to extract
    N = max(int(K * percent_frames), 1)  # Ensure at least one frame is extracted

    print(f"Sampling {N} out of {K} frames")

    # Generate indices of frames to extract
    indices = np.linspace(0, K - 1, N, dtype=int)
    indices_set = set(indices)

    current_frame = 0
    extracted_count = 0


    buffer_coords=[]
    buffer_color_vals=[]
    buffer_color_names=[]


    # intended for beautiful print terminal progress
    with Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
        ) as progress:
            task = progress.add_task("", total=K)


            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                progress.update(task, advance=1)


                frame_coords=[]
                frame_color_vals=[]
                frame_color_names=[]

                if current_frame in indices_set:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_ref = np.copy(frame)
                    coord_list=detect(frame, human_detector)
                    original_image = Image.fromarray(frame)
                    reference_image = Image.fromarray(frame_ref)
                    draw = ImageDraw.Draw(original_image) 
                    for coord in coord_list:
                        x1, y1, x2, y2 = coord
                        cropped_img = reference_image.crop((x1, y1, x2, y2))
                        try:
                            # PLAN -A : Do hard thresholding on otsu in post processing
                            color_name, color_val, ratio = segmentor(cropped_img, cloth_processor, cloth_segmenter, do_postprocess=True, thr_level='hard', device=device)

                            if ratio < 0.1:
                                # PLAN -B : Do soft thresholding on otsu in post processing, and it fails in ratio criteria
                                # do adaptive threshold in otsu
                                color_name, color_val, ratio = segmentor(cropped_img, cloth_processor, cloth_segmenter, do_postprocess=True, thr_level='soft', device=device)

                                if ratio < 0.1:
                                    color_name, color_val, ratio = segmentor(cropped_img, cloth_processor, cloth_segmenter, do_postprocess=True, thr_level='adaptive', device=device)
                        except:
                            # there are three possibilities of exception
                            # 1. the final mask of otsu (hard thresholded) has no positive values
                            # 2. the segmentation mask by segmentation model has few positive values, so post processing will be remove these values
                            # 3. the segmentation mask by segmentation model has no positive values (segmentation failed at that point)
                            try:
                                # PLAN -B : Do soft thresholding on otsu in post processing, and it fails in ratio criteria
                                # do adaptive threshold in otsu
                                color_name, color_val, ratio = segmentor(cropped_img, cloth_processor, cloth_segmenter, do_postprocess=True, thr_level='soft', device=device)
                                
                                if ratio < 0.1:
                                    color_name, color_val, ratio = segmentor(cropped_img, cloth_processor, cloth_segmenter, do_postprocess=True, thr_level='adaptive', device=device)
                            except:
                                # there are two possibilities of exception
                                # 1. the segmentation mask by segmentation model has few positive values, so post processing will be remove these values
                                # 2. the segmentation mask by segmentation model has no positive values (segmentation failed at that point)
                                try:
                                    color_name, color_val, ratio = segmentor(cropped_img, cloth_processor, cloth_segmenter, do_postprocess=False , thr_level='hard', device=device)

                                except:
                                    # there is only one possibility of exception
                                    # 1. the segmentation mask by segmentation model has no positive values (segmentation failed at that point)

                                    # PLAN-C : Search from buffer for similar colors in previous frames
                                    color_name, color_val = "unknown", (255, 255, 255)

                                    # If color is unknown, check buffer for similar coordinates
                                    if len(buffer_coords) > 0:
                                        # Calculate center of current detection
                                        current_center = (
                                            (x1 + x2) / 2,
                                            (y1 + y2) / 2
                                        )

                                        current_threshold = min(
                                            [abs(x2 - x1), abs(y2 - y1)]
                                            )
                                        
                                        min_distance = float('inf')
                                        best_color_name = "unknown"
                                        best_color_val = (255, 255, 255)
                                        
                                        # Check each frame in buffer (from recent to old)
                                        for frame_idx in range(len(buffer_coords) - 1, -1, -1):
                                            buff_coords = buffer_coords[frame_idx]
                                            buff_names = buffer_color_names[frame_idx]
                                            buff_vals = buffer_color_vals[frame_idx]
                                            
                                            # Check each detection in the frame
                                            for i, (bx1, by1, bx2, by2) in enumerate(buff_coords):
                                                if buff_names[i] != "unknown":
                                                    # Calculate center of buffer detection
                                                    buff_center = (
                                                        (bx1 + bx2) / 2,
                                                        (by1 + by2) / 2
                                                    )
                                                    
                                                    # Calculate Euclidean distance
                                                    distance = np.sqrt(
                                                        (current_center[0] - buff_center[0]) ** 2 +
                                                        (current_center[1] - buff_center[1]) ** 2
                                                    )
                                                    
                                                    # Update if this is the closest match so far
                                                    if distance < min_distance:
                                                        min_distance = distance
                                                        best_color_name = buff_names[i]
                                                        best_color_val = buff_vals[i]
                                        
                                        # If found a close match within current threshold, use it
                                        if min_distance <= current_threshold:
                                            color_name = best_color_name
                                            color_val = best_color_val





                        # update frame records
                        frame_coords.append((x1, y1, x2, y2))
                        frame_color_vals.append(color_val)
                        frame_color_names.append(color_name)



                        # set specifically for linux (for windows and mac, this will not work and would require different font type)
                        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
                        draw.rectangle([x1, y1, x2, y2], outline=color_val, width=6)
                        draw.text((x1, y1-20), color_name, fill=color_val, font=font)


                    # buffer update
                    buffer_coords.append(frame_coords)
                    buffer_color_vals.append(frame_color_vals)
                    buffer_color_names.append(frame_color_names)

                    # buffer size restriction
                    if len(buffer_coords) > 5:
                        buffer_coords.pop(0)
                        buffer_color_vals.pop(0)
                        buffer_color_names.pop(0)

                    # Save frame as image
                    img_name = f"img_{extracted_count}.png"
                    img_path = os.path.join(output_pth, img_name)
                    original_image.save(img_path)
                    extracted_count += 1

                current_frame += 1
                

            cap.release()


    print("Saving video as output.mp4...")
    save_video(output_pth, fps=fps)

    print("Openning video...")
    open_video("output.mp4")





if __name__ == "__main__":
    #ref: https://docs.ultralytics.com/models/yolo11/#performance-metrics
    human_detector = YOLO("yolo11s.pt")  # using this pretrained yolov11 model as it is the best balance of speed and mAP
    

    cloth_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    cloth_segmenter = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    human_detector = human_detector.to(device)
    cloth_segmenter = cloth_segmenter.to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_pth", type=str, default="input_video.mp4", help="path to video file")
    parser.add_argument("--output_pth", type=str, default="output_images", help="path to output directory")

    parser.add_argument("--percent_frames", type=float, default=1.0, help="percentage of frames to extract")

    parser.add_argument("--fps", type=int, default=10, help="fps of output video")
    args = parser.parse_args()

    main(args.video_pth, args.output_pth, args.percent_frames, human_detector, cloth_processor, cloth_segmenter, args.fps, device)
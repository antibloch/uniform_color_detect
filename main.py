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
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from detector import *
from segmentor import *


def open_video(pth):
    cap = cv2.VideoCapture(pth)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")

    # Read until video is completed
    while(cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            cv2.imshow('Frame', frame)
            
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()




def save_video(img_pth, fps=24):
   img_list=os.listdir(img_pth)
   img_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
   img_list = [os.path.join(img_pth, img) for img in img_list]
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   
   # Read first image to get dimensions
   first_img = cv2.imread(img_list[0])
   height, width = first_img.shape[:2]
   
   # Create video writer object
   out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
   
   # Write each image to video
   for img_path in img_list:
       frame = cv2.imread(img_path)
       out.write(frame)
   
   # Release video writer
   out.release()
   


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
    # intended for beautiful terminal progress
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

                if current_frame in indices_set:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    coord_list=detect(frame, human_detector)
                    original_image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(original_image) 
                    for coord in coord_list:
                        x1, y1, x2, y2 = coord
                        
                        cropped_img = original_image.crop((x1, y1, x2, y2))
                        try:
                            color_name, color_val = segmentor(cropped_img, cloth_processor, cloth_segmenter, True, device)
                        except:
                            try:
                                color_name, color_val = segmentor(cropped_img, cloth_processor, cloth_segmenter, False, device)
                            except:
                                color_name, color_val = "unknown", (255, 255, 255)

                        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
                        draw.rectangle([x1, y1, x2, y2], outline=color_val, width=6)
                        draw.text((x1, y1-20), color_name, fill=color_val, font=font)

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

    human_detector = YOLO("yolo11s.pt")  # using this model as it is best balance of speed and mAP
    cloth_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    cloth_segmenter = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    human_detector = human_detector.to(device)
    cloth_segmenter = cloth_segmenter.to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_pth", type=str, default="input_video.mp4", help="path to video file")
    parser.add_argument("--output_pth", type=str, default="output_images", help="path to output directory")

    parser.add_argument("--percent_frames", type=float, default=0.1, help="percentage of frames to extract")

    parser.add_argument("--fps", type=int, default=1, help="fps of output video")
    args = parser.parse_args()

    main(args.video_pth, args.output_pth, args.percent_frames, human_detector, cloth_processor, cloth_segmenter, args.fps, device)
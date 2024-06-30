import os
import cv2
import numpy as np
from pytube import YouTube
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from collections import defaultdict
import matplotlib.pyplot as plt

def download_youtube_video(url, output_path='video.mp4'):
    print("Downloading video from YouTube...")
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream.download(filename=output_path)
    print("Download complete.")
    return output_path

def extract_frames(video_path, output_folder='frames', frame_size=(224, 224), fps=1):
    print("Extracting frames from video...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    count = 0
    extracted_count = 0
    
    while count < total_frames:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, image = vidcap.read()
        if not success:
            break
        
        # Resize frame
        resized_image = cv2.resize(image, frame_size)
        # Normalize frame (values between 0 and 1)
        normalized_image = resized_image / 255.0
        # Save frame as JPEG file
        frame_path = os.path.join(output_folder, f"frame_{extracted_count:05d}.jpg")
        cv2.imwrite(frame_path, (normalized_image * 255).astype(np.uint8))
        count += frame_interval
        extracted_count += 1

        print(f"Extracted frame {extracted_count}/{total_frames//frame_interval}")

    vidcap.release()
    
    if extracted_count > 0:
        print(f"Video successfully processed into {extracted_count} frames with dimensions {frame_size}")
    else:
        print("Failed to process the video into frames")

def detect_humans_in_frame(frame, model, device):
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        detections = model(frame_tensor)
    return detections[0]

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def extract_and_track_humans(frames_folder, model, device, iou_threshold=0.5):
    print("Detecting and tracking humans in frames...")
    human_tracks = defaultdict(list)
    frames_files = sorted(os.listdir(frames_folder))
    prev_frame_humans = []

    for frame_index, frame_file in enumerate(frames_files):
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = Image.open(frame_path).convert("RGB")
            detections = detect_humans_in_frame(frame, model, device)
            boxes = detections['boxes']
            scores = detections['scores']
            labels = detections['labels']

            current_frame_humans = []
            for i in range(len(labels)):
                if labels[i] == 1 and scores[i] > 0.5:  # Assuming label 1 is for 'person'
                    current_frame_humans.append(boxes[i].cpu().numpy())
            
            # Associate current_frame_humans with prev_frame_humans
            for prev_human in prev_frame_humans:
                best_iou = 0
                best_match = None
                for i, curr_human in enumerate(current_frame_humans):
                    current_iou = iou(prev_human, curr_human)
                    if current_iou > best_iou and current_iou > iou_threshold:
                        best_iou = current_iou
                        best_match = i
                
                if best_match is not None:
                    human_tracks[id(prev_human)].append((frame_index, *current_frame_humans[best_match]))
                    current_frame_humans.pop(best_match)
                else:
                    human_tracks[id(prev_human)].append((frame_index, *prev_human))
            
            for human in current_frame_humans:
                human_tracks[id(human)].append((frame_index, *human))
            
            prev_frame_humans = current_frame_humans
            
            print(f"Processed frame {frame_index + 1}/{len(frames_files)}")

    return human_tracks

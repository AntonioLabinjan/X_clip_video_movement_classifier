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

# do tu smo sigurno dobri
def tracks_to_tensors(human_tracks, num_frames, frames_folder, frame_size=(224, 224)):
    print("Converting tracks to tensors...")
    human_tensors = {}
    frames_files = sorted(os.listdir(frames_folder))
    
    for human_id, track in human_tracks.items():
        track_array = np.zeros((num_frames, 4), dtype=np.float32)
        for entry in track:
            frame_index, x1, y1, x2, y2 = entry
            track_array[frame_index] = [x1, y1, x2, y2]
        human_tensors[human_id] = torch.tensor(track_array)

    # Now preprocess the frames for XCLIP
    xclip_human_tensors = {}
    for human_id, tensor in human_tensors.items():
        cropped_frames = []
        for frame_index, box in enumerate(tensor.numpy()):
            frame_path = os.path.join(frames_folder, frames_files[frame_index])
            frame = Image.open(frame_path).convert("RGB")
            if not np.array_equal(box, np.zeros(4)):  # If the box is not all zeros
                xmin, ymin, xmax, ymax = map(int, box)
                cropped_frame = frame.crop((xmin, ymin, xmax, ymax))
                cropped_frame = cropped_frame.resize(frame_size, Image.Resampling.LANCZOS)
                cropped_frame = np.array(cropped_frame) / 255.0  # Normalize to [0, 1]
                cropped_frames.append(cropped_frame)
            else:
                # If no bounding box, append a zero array or a blank frame of the same size
                cropped_frames.append(np.zeros((*frame_size, 3)))

        frames_array = np.array(cropped_frames, dtype=np.float32)
        xclip_human_tensors[human_id] = torch.tensor(frames_array).permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
    
    return xclip_human_tensors

# Example usage
video_url = 'https://www.youtube.com/watch?v=ORrrKXGx2SE'
video_path = 'video.mp4'
frames_output_folder = 'frames'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)

try:
    # Download the video
    downloaded_video_path = download_youtube_video(video_url, video_path)
    
    # Extract frames from the video
    extract_frames(downloaded_video_path, frames_output_folder)
    
    # Detect and track humans in the frames
    human_tracks = extract_and_track_humans(frames_output_folder, model, device)
    
    # Convert human movements to separate tensors and preprocess for XCLIP
    num_frames = len(os.listdir(frames_output_folder))
    xclip_human_tensors = tracks_to_tensors(human_tracks, num_frames, frames_output_folder)
    
    for human_id, tensor in xclip_human_tensors.items():
        print(f"Human ID {human_id} tensor shape: {tensor.shape}")
    print("Tensors created successfully for each human.")
except Exception as e:
    print(f"An error occurred: {e}")

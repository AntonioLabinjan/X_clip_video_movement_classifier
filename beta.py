# STEPS:
# Popravit brzinu videa da bude kao original
# Napravit bounding box oko čovika
# Shendlat rad s više ljudi

import os
import cv2
import numpy as np
from pytube import YouTube
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import XCLIPModel, XCLIPProcessor
from google.colab import files  # To download files from Colab

# Download video from YouTube
def download_youtube_video(url, output_path='video.mp4'):
    print("Downloading video from YouTube...")
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream.download(filename=output_path)
    print("Download complete.")
    return output_path

# Extract frames from video
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
        
        # Resize 
        resized_image = cv2.resize(image, frame_size)
        # Normalization to interval [0, 1]
        normalized_image = resized_image / 255.0
        # Save as JPG
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

# Detect humans in a frame
def detect_humans_in_frame(frame, model, device):
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        detections = model(frame_tensor)
    return detections[0]

# Track the single human across frames with smoothing
def track_single_human(frames_folder, model, device, iou_threshold=0.5, smooth_factor=0.8, output_folder='output_frames'):
    print("Detecting and tracking human in frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_files = sorted(os.listdir(frames_folder))
    human_track = []

    prev_human = None

    for frame_index, frame_file in enumerate(frames_files):
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = Image.open(frame_path).convert("RGB")
            detections = detect_humans_in_frame(frame, model, device)
            boxes = detections['boxes']
            scores = detections['scores']
            labels = detections['labels']

            current_human = None
            for i in range(len(labels)):
                if labels[i] == 1 and scores[i] > 0.5:  # Label 1 is for person
                    current_human = boxes[i].cpu().numpy()
                    break  # Assuming there's only one human in the video

            if prev_human is not None and current_human is not None:
                if iou(prev_human, current_human) < iou_threshold:
                    current_human = prev_human

            if current_human is not None:
                if prev_human is not None:
                    current_human = smooth_factor * prev_human + (1 - smooth_factor) * current_human
                human_track.append((frame_index, *current_human))
                prev_human = current_human

            # Draw bounding box on the frame
            frame_np = np.array(frame)
            if current_human is not None:
                x1, y1, x2, y2 = map(int, current_human)
                cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = "Person"
                cv2.putText(frame_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            output_frame_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

            print(f"Processed frame {frame_index + 1}/{len(frames_files)}")

    return human_track


# Convert track to tensor
def track_to_tensor(human_track, num_frames, frames_folder, frame_size=(224, 224), num_required_frames=8):
    print("Converting track to tensor...")
    frames_files = sorted(os.listdir(frames_folder))
    track_array = np.zeros((num_frames, 4), dtype=np.float32)

    for entry in human_track:
        frame_index, x1, y1, x2, y2 = entry
        track_array[frame_index] = [x1, y1, x2, y2]

    cropped_frames = []
    for frame_index, box in enumerate(track_array):
        frame_path = os.path.join(frames_folder, frames_files[frame_index])
        frame = Image.open(frame_path).convert("RGB")
        if not np.array_equal(box, np.zeros(4)):  # If the box is not all zeros
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_frame = frame.crop((xmin, ymin, xmax, ymax))
            cropped_frame = cropped_frame.resize(frame_size, Image.Resampling.LANCZOS)
            cropped_frame = np.array(cropped_frame) / 255.0  # Normalize to [0, 1]
            cropped_frames.append(cropped_frame)
        else:
            cropped_frames.append(np.zeros((*frame_size, 3)))

    if len(cropped_frames) < num_required_frames:
        padding = [np.zeros((*frame_size, 3))] * (num_required_frames - len(cropped_frames))
        cropped_frames.extend(padding)
    else:
        cropped_frames = cropped_frames[:num_required_frames]
    
    frames_array = np.array(cropped_frames, dtype=np.float32)
    return torch.tensor(frames_array).permute(0, 3, 1, 2)  # Convert to (num_frames, channels, height, width)

# Classify human movement using X-CLIP
def classify_human_movement(human_tensor, xclip_model, processor, device, frames_folder, output_folder):
    print("Classifying human movement...")
    classes = ["reading", "running", "jumping", "eating", "crawling", "walking", "sitting", "standing still"]

    # Prepare text inputs with special tokens and padding
    inputs = processor(
        text=[f"a person {action}" for action in classes],
        return_tensors="pt",
        padding=True
    )
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    human_tensor = human_tensor.to(device)
    
    with torch.no_grad():
        human_tensor = human_tensor.unsqueeze(0)  # Add batch dimension
        outputs = xclip_model(pixel_values=human_tensor, **inputs)
        logits = outputs.logits_per_text

        predictions = torch.argmax(logits, dim=0)
        predicted_class = classes[predictions.item()]

    # Add predicted class label to each frame
    frames_files = sorted(os.listdir(frames_folder))
    class_info = []
    for frame_index, frame_file in enumerate(frames_files):
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)

        label = f"Action: {predicted_class}"
        # Decrease font scale to make the text smaller
        font_scale = 0.3
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Position text in the top-left corner
        text_x = 10
        text_y = 30
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        # Add bounding box around the detected human
        if len(human_tensor.shape) == 4:  # Check if human_tensor is not empty
            x1, y1, x2, y2 = map(int, human_tensor.squeeze(0)[frame_index].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        output_frame_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_frame_path, frame)

        class_info.append((output_frame_path, (x1, y1, x2, y2) if len(human_tensor.shape) == 4 else None, predicted_class))

    print(f"Predicted class: {predicted_class}")
    return class_info

def create_output_video(frames_folder, output_video_path, class_info, fps=25):
    print("Creating output video...")
    frames_files = sorted(os.listdir(frames_folder))
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frames_files[0]))
    height, width, layers = first_frame.shape
    
    # Initialize video writer
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Loop through frames and write to video
    for frame_info in class_info:
        frame_path, bbox, predicted_class = frame_info
        frame = cv2.imread(frame_path)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add class label
        label = f"Action: {predicted_class}"
        font_scale = 0.3
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 30
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        video.write(frame)
    
    # Release video writer
    video.release()
    print(f"Output video created at {output_video_path}")
    return output_video_path  # Return the path of the created video

# Usage
video_url = 'https://youtu.be/xxrC2KGok08'
video_path = 'video.mp4'
frames_output_folder = 'frames'
output_frames_folder = 'output_frames'
output_video_path = 'output_video.mp4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)

try:
    # Download video
    downloaded_video_path = download_youtube_video(video_url, video_path)
    
    # Extract frames
    extract_frames(downloaded_video_path, frames_output_folder)
    
    # Track the single human in frames and save with bounding boxes
    human_track = track_single_human(frames_output_folder, model, device, output_folder=output_frames_folder)
    
    # Convert track to tensor
    num_frames = len(os.listdir(frames_output_folder))
    human_tensor = track_to_tensor(human_track, num_frames, frames_output_folder)

    # Load the X-CLIP model
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = xclip_model.to(device)

    # Classify human movement and save with class label and bounding box
    class_info = classify_human_movement(human_tensor, xclip_model, processor, device, frames_output_folder, output_frames_folder)
    
    # Create output video
    output_video_path = create_output_video(output_frames_folder, output_video_path, class_info)
    
    # Download the output video file
    print("Downloading the processed video...")
    files.download(output_video_path)
    
    print(f"The human in the video is {class_info[0][2]}")
    print("Movement classification and video creation completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

!pip install pytube
!pip install opencv-python-headless
!pip install numpy
!pip install pillow
!pip install torch torchvision
!pip install transformers

import os
import cv2
import numpy as np
from pytube import YouTube
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import XCLIPModel, XCLIPProcessor

# Download video from YouTube
def download_youtube_video(url, output_path='video.mp4'):
    print("Downloading video from YouTube...")
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream.download(filename=output_path)
    print("Download complete.")
    return output_path

# Extract frames from video with original FPS
def extract_frames(video_path, output_folder='frames'):
    print("Extracting frames from video...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {video_fps}")

    count = 0
    extracted_count = 0

    while count < total_frames:
        success, image = vidcap.read()
        if not success:
            break

        # Save as JPG
        frame_path = os.path.join(output_folder, f"frame_{extracted_count:05d}.jpg")
        cv2.imwrite(frame_path, image)
        count += 1
        extracted_count += 1

        print(f"Extracted frame {extracted_count}/{total_frames}")

    vidcap.release()

    if extracted_count > 0:
        print(f"Video successfully processed into {extracted_count} frames.")
    else:
        print("Failed to process the video into frames")

# Detect humans in a frame
def detect_humans_in_frame(frame, model, device):
    print("Detecting humans in frame...")
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        detections = model(frame_tensor)
    return detections[0]

# Assign unique IDs to detected humans based on bounding box IoU
def assign_ids_to_detections(previous_detections, current_detections, iou_threshold=0.5):
    if not previous_detections:
        # Initialize with current detections
        return {i: i for i in range(len(current_detections['boxes']))}

    id_mapping = {}
    assigned_ids = set()

    for i, current_box in enumerate(current_detections['boxes']):
        best_iou = 0
        best_id = None
        for j, previous_box in enumerate(previous_detections['boxes']):
            iou = compute_iou(previous_box, current_box)
            if iou > best_iou and iou > iou_threshold and j not in assigned_ids:
                best_iou = iou
                best_id = j

        if best_id is not None:
            id_mapping[i] = best_id
            assigned_ids.add(best_id)
        else:
            new_id = max(previous_detections['ids'].values(), default=-1) + 1
            id_mapping[i] = new_id
            previous_detections['ids'][new_id] = new_id

    return id_mapping

# Compute IoU between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_prime, y1_prime, x2_prime, y2_prime = box2

    inter_x1 = max(x1, x1_prime)
    inter_y1 = max(y1, y1_prime)
    inter_x2 = min(x2, x2_prime)
    inter_y2 = min(y2, y2_prime)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_prime - x1_prime) * (y2_prime - y1_prime)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Track multiple humans across frames
def track_multiple_humans(frames_folder, model, device, output_folder='output_frames'):
    print("Detecting and tracking humans in frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_files = sorted(os.listdir(frames_folder))
    human_tracks = {}
    previous_detections = {}

    for frame_index, frame_file in enumerate(frames_files):
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = Image.open(frame_path).convert("RGB")
            detections = detect_humans_in_frame(frame, model, device)

            if frame_index == 0:
                id_mapping = {i: i for i in range(len(detections['boxes']))}
            else:
                id_mapping = assign_ids_to_detections(previous_detections, detections)

            # Draw bounding box and assign IDs
            frame_np = np.array(frame)
            for i, box in enumerate(detections['boxes']):
                if detections['labels'][i] == 1 and detections['scores'][i] > 0.5:  # Label 1 is for person
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    id = id_mapping[i]
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_np, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    if id not in human_tracks:
                        human_tracks[id] = []
                    human_tracks[id].append(frame)

            previous_detections = {'boxes': detections['boxes'], 'ids': id_mapping}

            output_frame_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

            print(f"Processed frame {frame_index + 1}/{len(frames_files)}")

    return human_tracks

# Convert tracks to tensors
def convert_tracks_to_tensors(human_tracks, frame_size=(224, 224), num_required_frames=8):
    print("Converting tracks to tensors...")
    human_tensors = {}

    for track_id, frames in human_tracks.items():
        cropped_frames = []

        for frame in frames:
            frame = frame.resize(frame_size, Image.BILINEAR)  # Resize to required size
            cropped_frame = np.array(frame) / 255.0  # Normalize to [0, 1]
            cropped_frames.append(cropped_frame)

        # Handle padding for frames_array
        if len(cropped_frames) < num_required_frames:
            padding = [np.zeros((*frame_size, 3))] * (num_required_frames - len(cropped_frames))
            cropped_frames.extend(padding)
        else:
            cropped_frames = cropped_frames[:num_required_frames]

        frames_array = np.array(cropped_frames, dtype=np.float32)
        print(f"Shape of frames_array before permutation: {frames_array.shape}")

        # Convert to PyTorch tensor and permute dimensions
        frames_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2)  # (num_frames, channels, height, width)
        print(f"Shape of frames_tensor after permutation: {frames_tensor.shape}")

        human_tensors[track_id] = frames_tensor

    return human_tensors

# Classify human movement using X-CLIP
def classify_human_movement(human_tensors, xclip_model, processor, device):
    print("Classifying human movement...")
    classes = ["reading", "sleeping", "playing football", "eating", "crawling", "jumping"]
    movement_predictions = []

    # Prepare text inputs with special tokens and padding
    inputs = processor(
        text=classes,
        return_tensors="pt",
        padding=True
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    for track_id, human_tensor in human_tensors.items():
        human_tensor = human_tensor.to(device)
        with torch.no_grad():
            human_tensor = human_tensor.unsqueeze(0)  # Add batch dimension
            outputs = xclip_model(pixel_values=human_tensor, **inputs)
            logits = outputs.logits_per_text

            predictions = torch.argmax(logits, dim=0)
            predicted_class = classes[predictions.item()]
            movement_predictions.append((track_id, predicted_class))

    return movement_predictions

# Create video from frames
def create_video_from_frames(frames_folder, output_video_path, fps):
    print("Creating output video...")
    frames_files = sorted(os.listdir(frames_folder))

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frames_files[0]))
    height, width, layers = first_frame.shape

    # Initialize video writer
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Loop through frames and write to video
    for frame_file in frames_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    # Release video writer
    video.release()
    print(f"Output video created at {output_video_path}")
    return output_video_path  # Return the path of the created video

# Usage
video_url = 'https://www.youtube.com/shorts/bOBgAeebco0'
video_path = 'video.mp4'
frames_output_folder = 'frames'
output_frames_folder = 'output_frames'
output_video_path = 'output_video_with_boxes.mp4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)

try:
    # Download video
    downloaded_video_path = download_youtube_video(video_url, video_path)
    print(f"Video downloaded to {downloaded_video_path}")

    # Extract frames
    extract_frames(downloaded_video_path, frames_output_folder)
    print(f"Frames extracted to {frames_output_folder}")

    # Track multiple humans in frames and save with bounding boxes
    human_tracks = track_multiple_humans(frames_output_folder, model, device, output_folder=output_frames_folder)
    print("Human detection and bounding box drawing complete.")

    # Convert tracks to tensors
    human_tensors = convert_tracks_to_tensors(human_tracks)
    print("Human tracks converted to tensors.")

    # Load the X-CLIP model
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = xclip_model.to(device)
    print("X-CLIP model loaded.")

    # Classify human movement and save with class label
    movement_predictions = classify_human_movement(human_tensors, xclip_model, processor, device)
    print("Human movement classification complete.")

    for track_id, movement in movement_predictions:
        print(f"Track ID {track_id}: {movement}")

    fps = int(cv2.VideoCapture(downloaded_video_path).get(cv2.CAP_PROP_FPS))
    output_video_with_boxes = create_video_from_frames(output_frames_folder, output_video_path, fps)
    print(f"Output video with bounding boxes created at {output_video_with_boxes}")

except Exception as e:
    print(f"An error occurred: {e}")

# DOBIJEMO VIŠE BOXEVA ZA VIŠE LJUDI I ISPRINTA SE CLASS ZA SVAKU. JOŠ NAŠTIMAT ANNOTATION

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

# Track the single human across frames with smoothing
def track_single_human(frames_folder, model, device, output_folder='output_frames'):
    print("Detecting and tracking human in frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_files = sorted(os.listdir(frames_folder))

    for frame_index, frame_file in enumerate(frames_files):
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = Image.open(frame_path).convert("RGB")
            detections = detect_humans_in_frame(frame, model, device)
            boxes = detections['boxes']
            scores = detections['scores']
            labels = detections['labels']

            # Draw bounding box on the frame
            frame_np = np.array(frame)
            for i in range(len(labels)):
                if labels[i] == 1 and scores[i] > 0.5:  # Label 1 is for person
                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

            output_frame_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

            print(f"Processed frame {frame_index + 1}/{len(frames_files)}")

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

    # Track the single human in frames and save with bounding boxes
    track_single_human(frames_output_folder, model, device, output_folder=output_frames_folder)
    print("Human detection and bounding box drawing complete.")

    # Create output video with bounding boxes
    fps = int(cv2.VideoCapture(downloaded_video_path).get(cv2.CAP_PROP_FPS))
    output_video_with_boxes = create_video_from_frames(output_frames_folder, output_video_path, fps)
    print(f"Output video with bounding boxes created at {output_video_with_boxes}")
except Exception as e:
    print(f"An error occurred: {e}")

from transformers import XCLIPModel, XCLIPProcessor

# Convert video with bounding boxes back to frames
def extract_frames_from_video(video_path, output_folder='processed_frames'):
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

# Track to tensor conversion function
def track_to_tensor_with_boxes(frames_folder, num_frames, frame_size=(224, 224), num_required_frames=8):
    print("Converting frames with boxes to tensor...")
    frames_files = sorted(os.listdir(frames_folder))
    cropped_frames = []

    # Create cropped frames based on bounding boxes
    for frame_file in frames_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = Image.open(frame_path).convert("RGB")
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

    return frames_tensor

# Classify human movement using X-CLIP
def classify_human_movement(human_tensor, xclip_model, processor, device, frames_folder, output_folder):
    print("Classifying human movement...")
    classes = ["reading", "sleeping", "playing football", "eating", "crawling", "jumping"]

    # Prepare text inputs with special tokens and padding
    inputs = processor(
        text=classes,
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

        output_frame_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_frame_path, frame)

        class_info.append((output_frame_path, predicted_class))

    print(f"Predicted class: {predicted_class}")
    return class_info

# Create output video
def create_output_video_with_class(frames_folder, output_video_path, class_info, fps):
    print("Creating output video...")
    frames_files = sorted(os.listdir(frames_folder))

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frames_files[0]))
    height, width, layers = first_frame.shape

    # Initialize video writer
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Loop through frames and write to video
    for frame_info in class_info:
        frame_path, predicted_class = frame_info
        frame = cv2.imread(frame_path)

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
processed_frames_folder = 'processed_frames'
output_video_with_class_path = 'output_video_with_class.mp4'

try:
    # Extract frames from video with bounding boxes
    extract_frames_from_video(output_video_with_boxes, processed_frames_folder)
    print(f"Frames extracted to {processed_frames_folder}")

    # Convert track to tensor
    num_frames = len(os.listdir(processed_frames_folder))
    human_tensor = track_to_tensor_with_boxes(processed_frames_folder, num_frames)
    print("Human track converted to tensor.")

    # Load the X-CLIP model
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = xclip_model.to(device)
    print("X-CLIP model loaded.")

    # Classify human movement and save with class label
    class_info = classify_human_movement(human_tensor, xclip_model, processor, device, processed_frames_folder, output_frames_folder)
    print("Human movement classification complete.")

    # Create output video with class labels
    fps = int(cv2.VideoCapture(output_video_with_boxes).get(cv2.CAP_PROP_FPS))
    output_video_path = create_output_video_with_class(output_frames_folder, output_video_with_class_path, class_info, fps)

    print(f"The human in the video is {class_info[0][1]}")
    print("Movement classification and video creation completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

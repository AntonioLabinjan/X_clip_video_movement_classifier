!pip install torch torchvision transformers yt-dlp opencv-python-headless Pillow tqdm


import os
import cv2
import numpy as np
import yt_dlp
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from transformers import XCLIPModel, XCLIPProcessor
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to download video from YouTube
def download_youtube_video(url, output_path='video.mp4'):
    logger.info("Downloading video from YouTube...")
    ydl_opts = {'outtmpl': output_path, 'format': 'mp4'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info("Download complete.")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        raise

# Function to extract frames from video
def extract_frames(video_path, output_folder='frames'):
    logger.info("Extracting frames from video...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        logger.error(f"Error opening video file {video_path}")
        raise Exception(f"Error opening video file {video_path}")

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Total frames: {total_frames}, FPS: {video_fps}")

    count = 0
    extracted_count = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while count < total_frames:
            success, image = vidcap.read()
            if not success:
                break

            frame_path = os.path.join(output_folder, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_path, image)
            count += 1
            extracted_count += 1
            pbar.update(1)

    vidcap.release()
    logger.info(f"Video successfully processed into {extracted_count} frames.")
    return extracted_count, video_fps

# Function to detect humans in a frame
def detect_humans_in_frame(frame, model, device):
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        detections = model(frame_tensor)
    return detections[0]

# Function to assign unique IDs to detected humans
def assign_ids_to_detections(previous_detections, current_detections, previous_id_map, iou_threshold=0.5):
    if not previous_detections:
        current_id_map = {i: i for i in range(len(current_detections['boxes']))}
        return current_id_map

    current_id_map = {}
    assigned_ids = set()

    for i, current_box in enumerate(current_detections['boxes']):
        best_iou = 0
        best_id = None
        for j, previous_box in enumerate(previous_detections['boxes']):
            iou = compute_iou(previous_box.cpu().numpy(), current_box.cpu().numpy())
            if iou > best_iou and iou > iou_threshold and previous_id_map[j] not in assigned_ids:
                best_iou = iou
                best_id = previous_id_map[j]

        if best_id is not None:
            current_id_map[i] = best_id
            assigned_ids.add(best_id)
        else:
            new_id = max(previous_id_map.values(), default=-1) + 1
            current_id_map[i] = new_id
            assigned_ids.add(new_id)

    return current_id_map

# Function to compute IoU between two bounding boxes
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

# Function to track multiple humans across frames
def track_multiple_humans(frames_folder, model, device, output_folder='output_frames'):
    logger.info("Detecting and tracking humans in frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_files = sorted(os.listdir(frames_folder))
    human_tracks = {}
    previous_detections = {}
    previous_id_map = {}

    with tqdm(total=len(frames_files), desc="Tracking humans") as pbar:
        for frame_index, frame_file in enumerate(frames_files):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(frames_folder, frame_file)
                frame = Image.open(frame_path).convert("RGB")
                detections = detect_humans_in_frame(frame, model, device)

                if frame_index == 0:
                    id_mapping = {i: i for i in range(len(detections['boxes']))}
                else:
                    id_mapping = assign_ids_to_detections(previous_detections, detections, previous_id_map)

                frame_np = np.array(frame)
                for i, box in enumerate(detections['boxes']):
                    if detections['labels'][i] == 1 and detections['scores'][i] > 0.5:  # Label 1 is for person
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())
                        id = id_mapping[i]
                        cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        if id not in human_tracks:
                            human_tracks[id] = []
                        human_tracks[id].append((frame_file, (x1, y1, x2, y2)))

                previous_detections = {'boxes': detections['boxes'], 'ids': id_mapping}
                previous_id_map = id_mapping

                output_frame_path = os.path.join(output_folder, frame_file)
                cv2.imwrite(output_frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                pbar.update(1)

    logger.info("Human tracking complete.")
    return human_tracks

# Function to convert tracks to tensors
def convert_tracks_to_tensors(human_tracks, frames_output_folder, frame_size=(224, 224), num_required_frames=8):
    logger.info("Converting tracks to tensors...")
    human_tensors = {}

    for track_id, frames in human_tracks.items():
        cropped_frames = []

        for frame_file, box in frames:
            frame = Image.open(os.path.join(frames_output_folder, frame_file)).convert("RGB")
            frame = frame.resize(frame_size, Image.BILINEAR)  # Resize to required size
            cropped_frame = np.array(frame) / 255.0  # Normalize to [0, 1]
            cropped_frames.append(cropped_frame)

        if len(cropped_frames) < num_required_frames:
            padding = [np.zeros((*frame_size, 3))] * (num_required_frames - len(cropped_frames))
            cropped_frames.extend(padding)
        else:
            cropped_frames = cropped_frames[:num_required_frames]

        frames_array = np.array(cropped_frames, dtype=np.float32)
        frames_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2)  # (num_frames, channels, height, width)
        human_tensors[track_id] = frames_tensor

    logger.info("Track conversion to tensors complete.")
    return human_tensors

# Function to classify human movement using X-CLIP
def classify_human_movement(human_tensors, xclip_model, processor, device):
    logger.info("Classifying human movement...")
    classes = ["walking", "standing still", "running"]
    movement_predictions = {}

    inputs = processor(text=classes, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    for track_id, human_tensor in human_tensors.items():
        human_tensor = human_tensor.to(device)
        with torch.no_grad():
            human_tensor = human_tensor.unsqueeze(0)  # Add batch dimension
            outputs = xclip_model(pixel_values=human_tensor, **inputs)
            logits = outputs.logits_per_text

            predictions = torch.argmax(logits, dim=0)
            predicted_class = classes[predictions.item()]
            movement_predictions[track_id] = predicted_class
            logger.info(f"Track ID {track_id} classified as {predicted_class}")

    logger.info("Human movement classification complete.")
    return movement_predictions

# Function to create video from frames with annotations
def create_video_from_frames_with_annotations(frames_folder, output_video_path, fps, movement_predictions, human_tracks, output_folder='annotated_frames'):
    logger.info("Creating output video with annotations...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_files = sorted(os.listdir(frames_folder))

    for frame_file in tqdm(frames_files, desc="Annotating frames"):
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            for track_id, movement in movement_predictions.items():
                for frame_info in human_tracks[track_id]:
                    if frame_info[0] == frame_file:
                        x1, y1, x2, y2 = frame_info[1]
                        cv2.putText(frame, f"ID{track_id} - {movement}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

            annotated_frame_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(annotated_frame_path, frame)

    first_frame_path = os.path.join(output_folder, frames_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        logger.error("Error reading the first frame for video creation.")
        raise Exception("Error reading the first frame for video creation.")

    frame_height, frame_width = first_frame.shape[:2]
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame_file in tqdm(frames_files, desc="Creating video"):
        if frame_file.endswith('.jpg'):
            annotated_frame_path = os.path.join(output_folder, frame_file)
            frame = cv2.imread(annotated_frame_path)
            if frame is None:
                continue
            video.write(frame)

    video.release()
    logger.info(f"Output video created at {output_video_path}")
    return output_video_path

# Usage
video_url = 'https://youtu.be/_l7XGiDuydg'
video_path = 'video.mp4'
frames_output_folder = 'frames'
output_frames_folder = 'output_frames'
output_video_path = 'output_video_with_boxes.mp4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn_v2(pretrained=True).to(device)

try:
    # Download video
    downloaded_video_path = download_youtube_video(video_url, video_path)
    logger.info(f"Video downloaded to {downloaded_video_path}")

    # Extract frames
    frame_count, video_fps = extract_frames(downloaded_video_path, frames_output_folder)
    logger.info(f"Frames extracted to {frames_output_folder}")

    # Track multiple humans in frames and save with bounding boxes
    human_tracks = track_multiple_humans(frames_output_folder, model, device, output_folder=output_frames_folder)
    logger.info("Human detection and bounding box drawing complete.")

    # Convert tracks to tensors
    human_tensors = convert_tracks_to_tensors(human_tracks, frames_output_folder)
    logger.info("Human tracks converted to tensors.")

    # Load the X-CLIP model
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = xclip_model.to(device)
    logger.info("X-CLIP model loaded.")

    # Classify human movement and save with class label
    movement_predictions = classify_human_movement(human_tensors, xclip_model, processor, device)
    logger.info("Human movement classification complete.")

    for track_id, movement in movement_predictions.items():
        logger.info(f"Track ID {track_id}: {movement}")

    output_video_with_boxes = create_video_from_frames_with_annotations(output_frames_folder, output_video_path, video_fps, movement_predictions, human_tracks)
    logger.info(f"Output video with bounding boxes created at {output_video_with_boxes}")

except Exception as e:
    logger.error(f"An error occurred: {e}")

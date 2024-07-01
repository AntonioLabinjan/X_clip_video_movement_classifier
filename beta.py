# Sad je problem u temu ča ne prati mene ča delan nego me svaki put vidi kao 
# da san drugi čovik => ne napravi jedinstveni tensor za mene, nego napravi tensor svaki put kad me detektira
# to iduće riješavan
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
from transformers import AutoProcessor, XCLIPModel, XCLIPProcessor

# Skinen video
def download_youtube_video(url, output_path='video.mp4'):
    print("Downloading video from YouTube...")
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream.download(filename=output_path)
    print("Download complete.")
    return output_path

# extractamo frameove, dimenzije 224x224 da stane u xclip
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
        # Normalizacija na interval od 0 do 1
        normalized_image = resized_image / 255.0
        # Spremanje u JPG
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

# detectamo ljude
def detect_humans_in_frame(frame, model, device):
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        detections = model(frame_tensor)
    return detections[0]

# tu stvaramo neki box za svakega čovika u frameu
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

# procesiramo ljudski movement i stavljamo u array
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
                if labels[i] == 1 and scores[i] > 0.5:  
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

# pretvaramo array u tensor
def tracks_to_tensors(human_tracks, num_frames, frames_folder, frame_size=(224, 224), num_required_frames=8):
    print("Converting tracks to tensors...")
    human_tensors = {}
    frames_files = sorted(os.listdir(frames_folder))

    for human_id, track in human_tracks.items():
        track_array = np.zeros((num_frames, 4), dtype=np.float32)
        for entry in track:
            frame_index, x1, y1, x2, y2 = entry
            track_array[frame_index] = [x1, y1, x2, y2]
        human_tensors[human_id] = torch.tensor(track_array)

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
                cropped_frames.append(np.zeros((*frame_size, 3)))

        if len(cropped_frames) < num_required_frames:
            padding = [np.zeros((*frame_size, 3))] * (num_required_frames - len(cropped_frames))
            cropped_frames.extend(padding)
        else:
            cropped_frames = cropped_frames[:num_required_frames]
        
        frames_array = np.array(cropped_frames, dtype=np.float32)
        xclip_human_tensors[human_id] = torch.tensor(frames_array).permute(0, 3, 1, 2)  # Convert to (num_frames, channels, height, width)
    
    return xclip_human_tensors

# feedamo stvar u xclip
# Ovo je malo čudno; mislin da se svaki detected human gleda kao unique instance, a ne kao neki movement
def classify_human_movements(xclip_human_tensors, xclip_model, processor, device):
    print("Classifying human movements...")
    classes = ["walking", "running", "jumping", "eating", "sitting"]
    movement_classifications = {}

    for human_id, tensor in xclip_human_tensors.items():
        print(f"Classifying movement for human ID {human_id} with tensor shape {tensor.shape}")

        # Prepare text inputs with special tokens and padding
        inputs = processor(
            text=[f"a person {action}" for action in classes],
            return_tensors="pt",
            padding=True
        )
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        tensor = tensor.to(device)
        
        with torch.no_grad():
            # Ensure the tensor is in the right shape for the model
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            print(f"Tensor shape after unsqueeze: {tensor.shape}")

            outputs = xclip_model(pixel_values=tensor, **inputs)
            logits = outputs.logits_per_text
            print(f"Logits shape: {logits.shape}")

            # Get the predicted class
            predictions = torch.argmax(logits, dim=0)
            predicted_class = classes[predictions.item()]
            print(f"Human ID {human_id} predicted class: {predicted_class}")

        movement_classifications[human_id] = predicted_class

    return movement_classifications

# Usage
video_url = 'https://www.youtube.com/watch?v=84lYjtCfIvY'
video_path = 'video.mp4'
frames_output_folder = 'frames'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)

try:
    # skidanje
    downloaded_video_path = download_youtube_video(video_url, video_path)
    
    # Extractanje
    extract_frames(downloaded_video_path, frames_output_folder)
    
    # Detekcija i tracking ljudi u frameovima
    human_tracks = extract_and_track_humans(frames_output_folder, model, device)
    
    # frameovi u tensore (to zafrkava)
    num_frames = len(os.listdir(frames_output_folder))
    xclip_human_tensors = tracks_to_tensors(human_tracks, num_frames, frames_output_folder)

    # Load the X-CLIP model
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    xclip_model = xclip_model.to(device)

    # Classify human movements
    movement_classifications = classify_human_movements(xclip_human_tensors, xclip_model, processor, device)
    
    for human_id, movement in movement_classifications.items():
        print(f"Human ID {human_id} is {movement}")
    print("Movement classification completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

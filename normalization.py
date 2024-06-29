# Normaliziramo rate na 1 fps
import os
import cv2
import numpy as np
from pytube import YouTube
from PIL import Image

def download_youtube_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream.download(filename=output_path)
    return output_path

def extract_frames(video_path, output_folder='frames', frame_size=(224, 224), fps=1):
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

    vidcap.release()
    
    if extracted_count > 0:
        print(f"Video successfully processed into {extracted_count} frames with dimensions {frame_size}")
    else:
        print("Failed to process the video into frames")

# Example usage
video_url = 'https://www.youtube.com/watch?v=C3DjXmUCcMQ'
video_path = 'video.mp4'
frames_output_folder = 'frames'

try:
    # Download the video
    downloaded_video_path = download_youtube_video(video_url, video_path)
    
    # Extract frames from the video
    extract_frames(downloaded_video_path, frames_output_folder)
except Exception as e:
    print(f"An error occurred: {e}")

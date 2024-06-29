# breaking video into frames with dimensions 224x224 for each (the input shape for x-clip)

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

def extract_frames(video_path, output_folder='frames', frame_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        # Resize frame
        resized_image = cv2.resize(image, frame_size)
        # Save frame as JPEG file
        frame_path = os.path.join(output_folder, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_path, resized_image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    
    if count > 0:
        print(f"Video successfully processed into {count} frames with dimensions {frame_size}")
    else:
        print("Failed to process the video into frames")

# Example usage
video_url = 'https://www.youtube.com/watch?v=ORrrKXGx2SE'
video_path = 'video.mp4'
frames_output_folder = 'frames'

try:
    # Download the video
    downloaded_video_path = download_youtube_video(video_url, video_path)
    
    # Extract frames from the video
    extract_frames(downloaded_video_path, frames_output_folder)
except Exception as e:
    print(f"An error occurred: {e}")

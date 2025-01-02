import cv2
import os
import shutil
import zipfile

def extract_frames(video_path, output_path, interval):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    frame_count = 0

    # Loop through the video
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        # Check if the frame is read correctly
        if not ret:
            break
        # Save frame every 'interval' frames
        if frame_count % interval == 0:
            cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)
        frame_count += 1
    cap.release()

def create_zip_folder(folder_path, zip_path):
    # Create a zip file containing the folder contents
    shutil.make_archive(zip_path, 'zip', folder_path)

# Define the paths
video_path = "D:\\5th year\\GP1\\Code\\IMG_E442\\IMG_E0442.MOV"
output_folder_path = "D:\\5th year\\GP1\\Code\\IMG_E0442_test"
zip_file_path = "D:\\5th year\\GP1\\Code\\IMG_E0442_test.zip"

# Create the output folder and extract frames
os.makedirs(output_folder_path, exist_ok=True)
extract_frames(video_path, output_folder_path, interval=70)  # Save every 70th frame

# Create the zip file
create_zip_folder(output_folder_path, zip_file_path)

# Provide a link to download the zip file
print(f"Zip file created: {zip_file_path}")


import cv2 # openCV
import os
import random # this is needed to randomly split the data into training and validation sets

# mount Google Drive to access video files
from google.colab import drive
drive.mount('/content/drive')


base_video_dir = '/content/drive/MyDrive/person_videos'

output_base_dir = '/content/drive/MyDrive/dataset'

TRAIN_SPLIT_RATIO = 0.8 # plan is to have 80 percent train, 20 percent validation sets
FRAME_SKIP_INTERVAL = 5 # save one frame every 5 frames

person_videos = {
    'pryce': os.path.join(base_video_dir, 'pryce'),
    'kenny': os.path.join(base_video_dir, 'kenny'),
    'ben': os.path.join(base_video_dir, 'ben'),
}

# Directory in Google Drive Setup
#-----------------------------------------------------------
# create main output folder
os.makedirs(output_base_dir, exist_ok=True)

# paths for train/valid sets
train_dir = os.path.join(output_base_dir, 'train')
validation_dir = os.path.join(output_base_dir, 'validation')

# create train/validation folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# person specific subfolders within train and validation
# HAVE TO ORGANIZE DATA in format PyTorach imagefolder expects
for person_name in person_videos.keys():
  os.makedirs(os.path.join(train_dir, f'person_{person_name}'), exist_ok=True)
  os.makedirs(os.path.join(validation_dir, f'person_{person_name}'), exist_ok=True)

print("directory structure created successfully.")

#---------------------------------------------------------

# Loop through each person in the dictionary above
for person, folder_path in person_videos.items():
    print(f"\nProcessing videos for {person}")
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') or f.endswith('.mov')] # video files need to end in .mp4 or .mov this is important because I had to convert some video files

    if not video_files:
        print(f"No video files found in {folder_path}.")
        continue

    # counter ensures every frame saved for person has unique ID across multiple vids
    person_frame_count = 0

    # go through each vid for current person
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
            continue

        video_frame_idx = 0 # tracks our position within the current video

        # move through the video frame by frame
        while True:
            ret, frame = cap.read() # reads one frame from the video.
            # ret will be set to false once end of video is reached.
            if not ret:
                break

            if video_frame_idx % FRAME_SKIP_INTERVAL == 0:
                # Randomly assign frame to train or validation set
                if random.random() < TRAIN_SPLIT_RATIO:
                    # Save to train folder
                    output_sub_dir = os.path.join(train_dir, f'person_{person}')
                    set_type = 'train'
                else:
                    # Save to validation folder
                    output_sub_dir = os.path.join(validation_dir, f'person_{person}')
                    set_type = 'validation'

                # unique filename created
                # ex. pryce_original_vid_frame_01102.jpg
                filename = os.path.join(output_sub_dir, f'{person}_{video_file[:-4]}_frame_{person_frame_count:05d}.jpg') # save as a jpg
                cv2.imwrite(filename, frame)
                person_frame_count += 1 # unique frame counter incremented

            video_frame_idx += 1 # move to next frame

        # release vid file after all frames are processed.
        cap.release()
        print(f'Finished processing {video_file} for {person}. Total frames saved for {person} so far: {person_frame_count}')

print("\ncomplete.")

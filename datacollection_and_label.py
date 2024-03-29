import os
import shutil
from tqdm import tqdm

# Function to organize and label data
def organize_and_label_data(input_directory, output_directory):
    # Get a list of behavior folders
    behaviors = os.listdir(input_directory)

    for behavior in behaviors:
        behavior_folder = os.path.join(input_directory, behavior)

        if os.path.isdir(behavior_folder):
            # Create a corresponding folder in the output directory
            output_behavior_folder = os.path.join(output_directory, behavior)
            os.makedirs(output_behavior_folder, exist_ok=True)

            # Move all audio files from the behavior folder to the output folder
            for audio_file in tqdm(os.listdir(behavior_folder), desc=f"Processing {behavior}"):
                audio_file_path = os.path.join(behavior_folder, audio_file)
                output_audio_path = os.path.join(output_behavior_folder, audio_file)

                # Move the file
                shutil.move(audio_file_path, output_audio_path)

# Replace 'input_directory' with the path where your audio files are organized by behavior
input_directory = r"C:\Users\91901\Desktop\New folder\input_wavfiles"

# Replace 'output_directory' with the desired path for the organized and labeled data
output_directory = r"C:\Users\91901\Desktop\New folder\organized_data"

# Organize and label the data
organize_and_label_data(input_directory, output_directory)

import os
import librosa
import numpy as np

def extract_features(file_path, sample_rate=44100, n_mfcc=13):
    try:
        # Load the audio file
        audio, _ = librosa.load(file_path, sr=sample_rate)

        # Extract Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Calculate the mean of each MFCC dimension
        mean_mfccs = np.mean(mfccs, axis=1)

        return mean_mfccs

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_and_save_features(input_directory, output_directory, sample_rate=44100, n_mfcc=13):
    for behavior_label in os.listdir(input_directory):
        behavior_input_path = os.path.join(input_directory, behavior_label)
        behavior_output_path = os.path.join(output_directory, behavior_label)

        os.makedirs(behavior_output_path, exist_ok=True)

        if os.path.isdir(behavior_input_path):
            behavior_features = []

            for filename in os.listdir(behavior_input_path):
                file_path = os.path.join(behavior_input_path, filename)

                # Extract features from the audio file
                features = extract_features(file_path, sample_rate=sample_rate, n_mfcc=n_mfcc)

                # Append the features to the list
                if features is not None:
                    behavior_features.append(features)

            # Save the features as a single NumPy array
            behavior_features = np.array(behavior_features)
            np.save(os.path.join(behavior_output_path, f'{behavior_label}_features.npy'), behavior_features)

# Replace 'organized_data_directory' with the path where your organized data is located
organized_data_directory = r"C:\Users\91901\Desktop\New folder\Test_samples\organized_test_samples"

# Replace 'extracted_features_directory' with the path where you want to store the extracted features
extracted_features_directory = r"C:\Users\91901\Desktop\New folder\Test_samples\extracted_test_samples"

# Extract features and save them as single NumPy arrays for each behavior label
extract_and_save_features(organized_data_directory, extracted_features_directory)


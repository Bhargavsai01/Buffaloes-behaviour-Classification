import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Function to extract features from audio files
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

# Function to preprocess data, balance classes, and save features
def preprocess_and_save_features(input_directory, output_directory):
    all_features = []
    all_labels = []

    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)

        # Extract features from the audio file
        features = extract_features(file_path)

        # Append the features to the list
        if features is not None:
            all_features.append(features)
            all_labels.append(filename)

    # Check if there are any features before proceeding
    if not all_features:
        print("No features extracted. Check your input directory.")
        return

    # Convert labels to numeric representation
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    # Balance classes using SMOTE with n_neighbors smaller than the number of samples in the minority class
    n_minority_samples = min(np.bincount(encoded_labels))
    if n_minority_samples > 0:
        smote = SMOTE(sampling_strategy='auto', n_neighbors=min(6, n_minority_samples - 1), random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(all_features, encoded_labels)

        # Save the features and labels after balancing
        np.save(os.path.join(output_directory, 'X.npy'), features_resampled)
        np.save(os.path.join(output_directory, 'y.npy'), labels_resampled)
    else:
        print("No minority samples found for SMOTE. Check your input data distribution.")

# Replace 'input_data_directory' with the path where your audio files are located
input_data_directory = r"C:\Users\91901\Desktop\New folder\organized_data"

# Replace 'extracted_features_directory' with the path where you want to store the extracted features
extracted_features_directory = r"C:\Users\91901\Desktop\New folder\extracted_features"

# Preprocess data, balance classes using SMOTE, and save features
preprocess_and_save_features(input_data_directory, extracted_features_directory)

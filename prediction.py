import os
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import StandardScaler

# Replace these paths with your actual paths
model_path = r"C:\Users\91901\Desktop\New folder\rf_model.pkl"
scaler_path = r"C:\Users\91901\Desktop\New folder\scaler.pkl"
wav_file_path = r"C:\Users\91901\Downloads\grazing.wav"

# Load the trained model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Dictionary mapping behavior labels to numeric codes
behavior_types = {
    0: 'mother_baby_shouting',
    1: 'baby-shoting-and-running-for-milk',
    2: 'babyshouting',
    3: 'fear',
    4: 'grazing',
    5: 'growling',
    6: 'hungry',
    7: 'other',
    8: 'owner',
    9: 'fear_running',
    10: 'running shout',
    11: 'mother_shouting',
    12: 'snore',
    13: 'unknown',
    14: 'sick',
    15: 'angry'
}

# Function to extract features from a new WAV file
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

# Extract features from the new WAV file
features = extract_features(wav_file_path)

# Check if features were successfully extracted
if features is not None:
    # Reshape features for prediction
    features = features.reshape(1, -1)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)

    # Make predictions using the trained model
    predicted_label = model.predict(scaled_features)

    # Map the numeric code to behavior type
    predicted_behavior = behavior_types[predicted_label[0]]

    print(f"Prediction for {wav_file_path}: {predicted_behavior}")

    # Print model's predictions for each class
    predictions_for_each_class = model.predict_proba(scaled_features)
    for label, prob in zip(model.classes_, predictions_for_each_class[0]):
        behavior = behavior_types[label]
        print(f"Probability for {behavior}: {prob}")

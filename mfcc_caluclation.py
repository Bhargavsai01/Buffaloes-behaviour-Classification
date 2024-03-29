import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
audio_file = r"C:\Users\91901\Desktop\New folder\organized_data\baby-shoting-and-running-for-milk\baby-shoting-and-running-for-milk_1.wav"
signal, sr = librosa.load(audio_file, sr=None)

print("Original Signal Shape:", signal.shape)

# Pre-emphasis
signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

print("Signal after Pre-emphasis Shape:", signal.shape)

# Compute the Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=512, hop_length=160, n_mels=40)

print("Mel Spectrogram Shape:", mel_spec.shape)

# Logarithm
log_mel_spec = np.log(mel_spec + 1e-10)

print("Log Mel Spectrogram Shape:", log_mel_spec.shape)

# DCT
mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=13)

print("MFCC Shape:", mfcc.shape)

# Keep the first 13 coefficients as the final MFCC features
mfcc_features = mfcc[1:14, :]

print("Final MFCC Features Shape:", mfcc_features.shape)

# Print extracted features
print("Original Signal:")
print(signal)

print("\nMel Spectrogram:")
print(mel_spec)

print("\nLog Mel Spectrogram:")
print(log_mel_spec)

print("\nMFCC Features:")
print(mfcc_features)

# Plot waveform at regular intervals
interval_duration = 1  # in seconds
num_intervals = int(len(signal) / sr / interval_duration)

plt.figure(figsize=(12, 6))

for i in range(num_intervals):
    start_sample = int(i * sr * interval_duration)
    end_sample = int((i + 1) * sr * interval_duration)
    
    plt.subplot(num_intervals, 1, i + 1)
    librosa.display.waveshow(signal[start_sample:end_sample], sr=sr, color='b')
    plt.title(f'Waveform at {i * interval_duration}s')

plt.tight_layout()
plt.show()

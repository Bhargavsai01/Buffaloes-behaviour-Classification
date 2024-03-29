import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # for saving the model

# Define the path where your extracted features are stored
extracted_features_directory = r"C:\Users\91901\Desktop\New folder\extracted_features"

# Load the preprocessed features and labels
features = np.load(os.path.join(extracted_features_directory, 'X.npy'))
labels = np.load(os.path.join(extracted_features_directory, 'y.npy'))

# Get the unique labels and their counts
unique_labels, label_counts = np.unique(labels, return_counts=True)

# Specify the minimum number of samples for each class
min_samples_per_class = 5

# Initialize empty arrays for the splits
X_train, X_valid, X_test = np.array([]), np.array([]), np.array([])
Y_train, Y_valid, Y_test = np.array([]), np.array([]), np.array([])

# Iterate over each class
for label in unique_labels:
    # Filter data for the current class
    class_indices = np.where(labels == label)[0]

    # Check if the class has enough samples for splitting
    if len(class_indices) < min_samples_per_class:
        print(f"Skipping class {label} due to insufficient data.")
        continue

    class_features = features[class_indices]
    class_labels = labels[class_indices]

    # Perform a 80-10-10 split for each class
    X_temp, X_test_class, Y_temp, Y_test_class = train_test_split(
        class_features, class_labels, test_size=0.1, random_state=42, stratify=class_labels
    )

    X_train_class, X_valid_class, Y_train_class, Y_valid_class = train_test_split(
        X_temp, Y_temp, test_size=0.1111, random_state=42, stratify=Y_temp
    )

    # Concatenate the splits to the overall splits
    X_train = np.vstack([X_train, X_train_class]) if X_train.size else X_train_class
    X_valid = np.vstack([X_valid, X_valid_class]) if X_valid.size else X_valid_class
    X_test = np.vstack([X_test, X_test_class]) if X_test.size else X_test_class
    Y_train = np.concatenate([Y_train, Y_train_class], axis=0)
    Y_valid = np.concatenate([Y_valid, Y_valid_class], axis=0)
    Y_test = np.concatenate([Y_test, Y_test_class], axis=0)

# Replace 'C:\Users\91901\Desktop\New folder\train_valid_test_split' with the desired output path
output_path = r"C:\Users\91901\Desktop\New folder\train_valid_test_split"

# Save the split datasets
np.save(os.path.join(output_path, 'X_train.npy'), X_train)
np.save(os.path.join(output_path, 'Y_train.npy'), Y_train)
np.save(os.path.join(output_path, 'X_valid.npy'), X_valid)
np.save(os.path.join(output_path, 'Y_valid.npy'), Y_valid)
np.save(os.path.join(output_path, 'X_test.npy'), X_test)
np.save(os.path.join(output_path, 'Y_test.npy'), Y_test)

print("Dataset split into training, validation, and test sets.")

# Now you can use X_train, Y_train, X_valid, Y_valid, X_test, Y_test for training and evaluation.

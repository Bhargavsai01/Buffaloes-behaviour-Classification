import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load your data or replace this with your data loading code
# For example, if you have saved X_train, X_test, y_train, y_test using train_test_split
train_test_split_directory = r"C:\Users\91901\Desktop\New folder\train_valid_test_split"
X_train = np.load(os.path.join(train_test_split_directory, 'X_train.npy'))
X_test = np.load(os.path.join(train_test_split_directory, 'X_test.npy'))
y_train = np.load(os.path.join(train_test_split_directory, 'y_train.npy'))
y_test = np.load(os.path.join(train_test_split_directory, 'y_test.npy'))

# Standardize the data (optional but can be beneficial for RandomForest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize RandomForest classifier with adjusted hyperparameters
random_forest_classifier = RandomForestClassifier(
    n_estimators=1000,  # Increase the number of trees
    max_depth=20,       # Adjust the maximum depth of the trees
    min_samples_split=2,  # Adjust the minimum samples required to split an internal node
    min_samples_leaf=1   # Adjust the minimum number of samples required to be at a leaf node
)

# Train the classifier
random_forest_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = random_forest_classifier.predict(X_test_scaled)

# Evaluate the model on the testing set
print("Testing Set:")
print(classification_report(y_test, y_pred, zero_division=1))  # Adjust zero_division here
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Save the trained model and scaler
model_path = r"C:\Users\91901\Desktop\New folder\codes\r_forest_model.pkl"
scaler_path = r"C:\Users\91901\Desktop\New folder\codes\s_scaler.pkl"

with open(model_path, 'wb') as model_file:
    pickle.dump(random_forest_classifier, model_file)

with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

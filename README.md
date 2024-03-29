Overview:

Buffalo Behaviour Classification is a project aimed at classifying different behaviors exhibited by buffaloes using machine learning techniques. This project focuses on analyzing audio recordings of buffalo sounds and predicting their behaviors in real-time.

Motivation:

Understanding buffalo behavior is crucial for various applications, including agriculture, veterinary science, and wildlife conservation. By accurately classifying buffalo behaviors, farmers and researchers can monitor their health, well-being, and environmental interactions more effectively.

Features:

Real-time prediction of buffalo behavior based on audio recordings
Integration with Twilio for sending SMS alerts with behavior predictions
Use of machine learning models for classification tasks
Dependencies
Flask: A micro web framework for Python
scikit-learn: A machine learning library for Python
librosa: A Python package for audio and music signal analysis      
output image:                           ![Uploading babyshouting_output.jpg…]()



Twilio: A cloud communications platform for building SMS and voice applications
Usage
Setup Environment: Install the required dependencies using pip install -r requirements.txt.
Run the Application: Execute the Flask application using python app.py. The application will start running on http://127.0.0.1:5000/.
Upload Audio File: Visit the provided URL and upload an audio file containing buffalo sounds.
View Prediction: The application will predict the behavior based on the uploaded audio and display the result along with prediction accuracy, message, and relevant links.

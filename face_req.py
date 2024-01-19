import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# Load the trained FaceNet model
facenet_model = load_model('facenet_model_custom.h5')

# Function to preprocess the face for recognition
def preprocess_face(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_pixels

# Function to load the dataset
def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            face = cv2.resize(img, (160, 160))
            face = preprocess_face(face)
            X.append(face)
            y.append(subdir)
    return np.asarray(X), np.asarray(y)

# Load the dataset
dataset_directory = 'C:/Users/pskar/OneDrive/Documents/GitHub/S3/dataset'
X, y = load_dataset(dataset_directory)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X.reshape(X.shape[0], -1), y_encoded)

# Save the trained SVM model and label encoder
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Function to recognize faces in real-time
def recognize_faces():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture video.")
            break

        faces = detector.detect_faces(frame)

        for result in faces:
            x, y, width, height = result['box']
            x, y = abs(x), abs(y)
            face = frame[y:y+height, x:x+width]
            face = cv2.resize(face, (160, 160))
            face = preprocess_face(face)
            face_embedding = facenet_model.predict(np.expand_dims(face, axis=0))[0]

            face_embedding = face_embedding.reshape(1, -1)
            face_embedding = preprocess_input(face_embedding)

            prediction = svm_model.predict(face_embedding)
            confidence = np.max(svm_model.predict_proba(face_embedding))

            person = label_encoder.inverse_transform(prediction)[0]

            if confidence > 0.7:
                text = f"{person} (Confidence: {confidence:.2f})"
                color = (0, 255, 0)  # Green color for bounding box
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Red color for bounding box

            cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)  # Draw bounding box
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Display name and confidence

        cv2.imshow('Facial Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time facial recognition
recognize_faces()

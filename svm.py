import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from mtcnn.mtcnn import MTCNN
import joblib  # Import joblib directly

# Function to preprocess the face for recognition
def preprocess_face(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_pixels

# Function to load and preprocess the dataset
def load_dataset(directory, detector):
    X, labels = [], []  # Changed the variable name from 'y' to 'labels'
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            faces = detector.detect_faces(img)
            
            for result in faces:
                x, y, width, height = result['box']
                x, y = abs(x), abs(y)
                face = img[y:y+height, x:x+width]
                face = cv2.resize(face, (160, 160))
                face = preprocess_face(face)
                X.append(face)
                labels.append(label)  # Changed the variable name from 'y' to 'labels'

    return np.array(X), np.array(labels)  # Changed the variable name from 'y' to 'labels'

# Load the FaceNet model
facenet_model = load_model('facenet_model_custom.h5')

# Create the MTCNN detector
detector = MTCNN()

# Load and preprocess the dataset
dataset_path = 'dataset'
X, labels = load_dataset(dataset_path, detector)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Extract facial embeddings using the FaceNet model
X_embeddings = []
for face in X:
    face_embedding = facenet_model.predict(np.expand_dims(face, axis=0))[0]
    X_embeddings.append(face_embedding)

X_embeddings = np.array(X_embeddings)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_encoded, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Save the SVM model and label encoder
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from mtcnn.mtcnn import MTCNN
import joblib

# Load the trained FaceNet model
facenet_model = load_model('facenet_model_custom.h5')

# Load the SVM model and label encoder
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to preprocess the face for recognition
def preprocess_face(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_pixels

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
            else:
                text = "Unknown"

            print(text)  # Output the result to the console

        #cv2.imshow('Facial Recognition', frame)

        #if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
          #  break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time facial recognition
recognize_faces()

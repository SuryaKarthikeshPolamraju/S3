import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# Function to load and preprocess images
def load_images(directory):
    images, labels = [], []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = Image.open(img_path).convert('RGB').resize((160, 160))
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
dataset_path = 'dataset'
X, y = load_images(dataset_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Preprocess images for FaceNet model
X_train_preprocessed = preprocess_input(X_train)
X_test_preprocessed = preprocess_input(X_test)

# Define a custom FaceNet model
def create_custom_facenet_model():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(160, 160, 3), pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output layer
    ])
    
    return model

# Create and compile the FaceNet model
facenet_model = create_custom_facenet_model()
facenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the FaceNet model
facenet_model.fit(X_train_preprocessed, y_train_encoded, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
y_pred = facenet_model.predict(X_test_preprocessed)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test_encoded, y_pred_classes)
print(f'Test Accuracy: {accuracy}')

# Save the FaceNet model
facenet_model.save('facenet_model_custom.h5')

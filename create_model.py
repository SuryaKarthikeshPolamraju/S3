import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load and preprocess images
def load_and_preprocess_data(data_directory):
    images = []
    labels = []

    # Iterate through each subdirectory (assuming each subdirectory represents a person)
    for person_name in os.listdir(data_directory):
        person_path = os.path.join(data_directory, person_name)
        
        # Ignore non-directory files
        if not os.path.isdir(person_path):
            continue

        # Label each person with a unique identifier
        label = len(labels)

        # Iterate through each image in the subdirectory
        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)

            # Read and preprocess the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (160, 160))  # Resize to a common size

            # Normalize pixel values to be between 0 and 1
            image = image / 255.0

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)

# Specify the path to your custom faces dataset
custom_faces_directory = 'dataset'

# Load and preprocess the data
images, labels = load_and_preprocess_data(custom_faces_directory)

# Create the FaceNet-like model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128)  # Adjust the output size based on your requirements
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10, batch_size=32)

# Save the model and weights to an HDF5 file
model.save('facenet_model.h5')


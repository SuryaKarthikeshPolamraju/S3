import h5py

# Replace 'facenet_keras.h5' with your actual model file
model_path = 'facenet_keras.h5'

try:
    with h5py.File(model_path, 'r'):
        print(f"The file '{model_path}' is a valid HDF5 file.")
except Exception as e:
    print(f"Error: {e}")

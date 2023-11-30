from matplotlib import pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import numpy as np
from PIL import Image
from numpy import asarray
from numpy import array
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from numpy import expand_dims
from numpy import reshape
from numpy import load
from numpy import max
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import time
import keyboard
vs = VideoStream(src=0,framerate=10).start()
time.sleep(2.0)
fps = FPS().start()
def extract_image(image):
  if isinstance(image, str):  # If the input is a file path
    img1 = Image.open(image)
  else:  # If the input is a NumPy array (camera frame)
    img1 = Image.fromarray(image, 'RGB')
  img1 = img1.convert('RGB')
  pixels = asarray(img1)
  detector = MTCNN()
  f = detector.detect_faces(pixels)
  x1, y1, w, h = f[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2 = abs(x1 + w)
  y2 = abs(y1 + h)
  store_face = pixels[y1:y2, x1:x2]
  image1 = Image.fromarray(store_face, 'RGB')
  image1 = image1.resize((160, 160))
  face_array = asarray(image1)
  return face_array

#extracting embeddings
def extract_embeddings(model,face_pixels):
  face_pixels = face_pixels.astype('float32')
  mean = face_pixels.mean()
  std  = face_pixels.std()
  face_pixels = (face_pixels - mean)/std
  samples = expand_dims(face_pixels,axis=0)
  yhat = model.predict(samples)
  return yhat[0]
  
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    face = extract_image(frame)
    testx = asarray(face)
    testx = testx.reshape(-1,160,160,3)
    #print("Input test data shape: ",testx.shape)

    #find embeddings
    model = load_model('facenet_model.h5')
    new_testx = list()
    for test_pixels in testx:
        embeddings = extract_embeddings(model,test_pixels)
        new_testx.append(embeddings)
    new_testx = asarray(new_testx)  
    #print("Input test embedding shape: ",new_testx.shape)

    data1 = load('student-dataset.npz')
    train_x,train_y = data1['arr_0'],data1['arr_1']

    data = load('student-embeddings.npz')
    trainx,trainy= data['arr_0'],data['arr_1']
    #print("Loaded data: Train=%d , Test=%d"%(trainx.shape[0],new_testx.shape[0]))

    #normalize the input data
    in_encode = Normalizer(norm='l2')
    trainx = in_encode.transform(trainx)
    new_testx = in_encode.transform(new_testx)

    #create a label vector
    new_testy = trainy 
    out_encode = LabelEncoder()
    out_encode.fit(trainy)
    trainy = out_encode.transform(trainy)
    new_testy = out_encode.transform(new_testy)

    #define svm classifier model 
    model =SVC(kernel='linear', probability=True)
    model.fit(trainx,trainy)

    #predict
    predict_train = model.predict(trainx)
    predict_test = model.predict(new_testx)

    #get the confidence score
    probability = model.predict_proba(new_testx)
    confidence = max(probability)

    #Accuracy
    acc_train = accuracy_score(trainy,predict_train)

    # ... (previous code)

    # Predicted label
    predicted_label = out_encode.inverse_transform(predict_test)[0]
    print("Predicted Label:", predicted_label)
    if keyboard.is_pressed('q'):
        print("The 'q' key is pressed. Exiting...")
        break
    
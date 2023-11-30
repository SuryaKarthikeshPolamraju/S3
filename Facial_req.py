from matplotlib import pyplot as plt
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

def extract_image(image):
  img1 = Image.open(image)
  img1 = img1.convert('RGB')
  pixels = asarray(img1)
  detector = MTCNN()
  f = detector.detect_faces(pixels)
  x1,y1,w,h = f[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2 = abs(x1+w)
  y2 = abs(y1+h)
  store_face = pixels[y1:y2,x1:x2]
  image1 = Image.fromarray(store_face,'RGB')
  image1 = image1.resize((160,160))
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
  
#load data and reshape the image
Img = 'Surya.jpg'
#plt.imshow(Img)
face = extract_image(Img)
testx = asarray(face)
testx = testx.reshape(-1,160,160,3)
print("Input test data shape: ",testx.shape)

#find embeddings
model = load_model('facenet_model.h5')
new_testx = list()
for test_pixels in testx:
  embeddings = extract_embeddings(model,test_pixels)
  new_testx.append(embeddings)
new_testx = asarray(new_testx)  
print("Input test embedding shape: ",new_testx.shape)

data1 = load('student-dataset.npz')
train_x,train_y = data1['arr_0'],data1['arr_1']

data = load('student-embeddings.npz')
trainx,trainy= data['arr_0'],data['arr_1']
print("Loaded data: Train=%d , Test=%d"%(trainx.shape[0],new_testx.shape[0]))

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


# Display Input Image
plt.subplot(1, 2, 1)
plt.imshow(face)
predicted_label = out_encode.inverse_transform(predict_test)[0]
plt.title("Predicted: {}".format(predicted_label))
plt.xlabel("Input Image")

# Display Predicted and True Labels with Confidence
plt.subplot(1, 2, 2)

# Check if the predicted label is in the training labels
if predicted_label in out_encode.classes_:
    val = np.argmax(probability)
    plt.imshow(train_x[val])
    true_label = out_encode.inverse_transform([train_y[val]])[0]
    confidence_percentage = confidence * 100
    plt.title("Predicted: {}\nTrue Label: {}\nConfidence: {:.2f}%".format(predicted_label, true_label, confidence_percentage))
    plt.xlabel("Predicted Data")
else:
    # Add the unseen label to the label encoder classes
    out_encode.classes_ = np.append(out_encode.classes_, predicted_label)

    plt.title("Predicted: {}\nTrue Label: Unseen label\nConfidence: {:.2f}%".format(predicted_label, confidence * 100))
    plt.xlabel("Predicted Data")

plt.tight_layout()
plt.show()
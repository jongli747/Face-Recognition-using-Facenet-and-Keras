import cv2
#from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from PIL import Image
from cv2 import rectangle
from cv2 import CascadeClassifier
from numpy import expand_dims
from FaceDetector import *

IMAGE_SIZE = 160
embeddings = np.load('face_embeddings.npy')
labels = np.load('labels.npy')

#print(embeddings.shape, labels.shape)

model = load_model('./model/facenet_keras.h5')
#print(model.summary())

def get_embedding(model, face_pixels):
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis = 0)
    #get embedding
    y0 = model.predict(samples)
#     print(y0.shape)
#     print(y0)
    return y0[0]

def distance(e1,e_org):
    return np.sum(np.square(e1 - e_org))

def inference(model, img, embeddings, labels):
    img_emb = get_embedding(model, img)
    print(img_emb.shape)
    min_distance = 1000
    for i in range(embeddings.shape[0]):
        dis = distance(img_emb, embeddings[i])
        if dis < min_distance and dis != 0.0:
            min_distance = dis
            who = labels[i]
    return who, min_distance
    #print(who, min_distance)

camera = cv2.VideoCapture(0)
detector = faceDetector('./model/haarcascade_frontalface_default.xml')

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame.shape)
    fd = detector.detect(gray)
    for (x,y,w,h) in fd:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi,(IMAGE_SIZE, IMAGE_SIZE))

        who, dist = inference(model, roi, embeddings, labels)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, who, (x+ (w//2), y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType = cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

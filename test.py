# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
def detect(img):
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 face_casc = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
 faces = face_casc.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
 if (len(faces) == 0):
  return None, None
 (x, y, w, h) = faces[0]
 return gray[y:y + w, x:x + h], faces[0]
def train(path):
 dirs = os.listdir(path)
 faces = []
 labels = []
 for name in dirs:
  label = int(name)
  ppath = path + "/" + name
  nnames = os.listdir(ppath)
  for nname in nnames:
   pppath = ppath + "/" + nname
   image = cv2.imread(pppath)
   face, rect = detect(image)
   if face is not None:
    faces.append(face)
    labels.append(label)
 return faces, labels
print("training...hhh")
faces, labels = train("training_data")
face_recog = cv2.face.LBPHFaceRecognizer_create()
face_recog.train(faces, np.array(labels))
def draw_rect(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)
def draw_text(img, text, x, y):
 cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)
names = ["zjs", "clw","zpn"]
def predict(test_img):
 img = test_img.copy()
 face, rect = detect(img)
 label = face_recog.predict(face)
 ttext = names[label[0]]
 draw_rect(img, rect)
 draw_text(img, ttext, rect[0], rect[1] - 5)
 return img
test_img1 = cv2.imread("test_data/test1.jpg")
test_img2 = cv2.imread("test_data/test2.jpg")
test_img3 = cv2.imread("test_data/test3.jpg")
pred_img1 = predict(test_img1)
pred_img2 = predict(test_img2)
pred_img3 = predict(test_img3)
cv2.destroyAllWindows()
cv2.imshow("test1", pred_img1)
cv2.imshow("test2", pred_img2)
cv2.imshow("test3", pred_img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
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
cap = cv2.VideoCapture(0)
while (1):
 ret, frame = cap.read()
 frame = cv2.flip(frame, 1)
 cv2.imshow("capture", frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
  cv2.imwrite("test.jpg", frame)
  break
 cap.release()
 cv2.destroyAllWindows()
print("training...")
faces, labels = train("training_data")
face_recog = cv2.face.LBPHFaceRecognizer_create()
face_recog.train(faces, np.array(labels))
def draw_rect(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
def draw_text(img, text, x, y):
 cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
names = ["zjs", "clw","zpn"]
def predict(test_img):
 img = test_img.copy()
 face, rect = detect(img)
 label = face_recog.predict(face)
 ttext = names[label[0]]
 draw_rect(img, rect)
 draw_text(img, ttext, rect[0], rect[1] - 5)
 return img
test_img = cv2.imread("test.jpg")
pred_img = predict(test_img)
cv2.imshow("test", pred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
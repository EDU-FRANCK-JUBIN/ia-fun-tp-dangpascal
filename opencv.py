""" Test 1
import cv2
import sys
from matplotlib import pyplot as plt

imagePath = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/image0.jpg'
dirCascadeFiles = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/opencv/data/haarcascades_cuda/'
cascadefile = dirCascadeFiles + "haarcascade_frontalface_default.xml"
classCascade = cv2.CascadeClassifier(cascadefile)
image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = classCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(image)
plt.show()
print("Il y a {0} visage(s).".format(len(faces)))
"""

""" Test 2
import cv2
import sys
from matplotlib import pyplot as plt
import time
import os
print(os.getcwd())

imagePath = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/image0.jpg'
dirCascadeFiles = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/opencv/data/haarcascades_cuda/'
cascadefile = dirCascadeFiles + "haarcascade_frontalface_alt.xml"
classCascade = cv2.CascadeClassifier(cascadefile)
image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = classCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

i=0
for (x, y, w, h) in faces:
    crop_img = image[y:y+h, x:x+w]
    cv2.imwrite('fichier_resultat_' + str(i) + '.jpg', image[y:y+h, x:x+w])
    i = i+1
plt.imshow(crop_img)
plt.show()
print("Il y a {0} visage(s).".format(len(faces)))
"""
""" Test 3
import cv2
import sys
from matplotlib import pyplot as plt

imagePath = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/image3.jpg'
dirCascadeFiles = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/opencv/data/haarcascades_cuda/'
cascadefile = dirCascadeFiles + "haarcascade_frontalface_default.xml"
classCascade = cv2.CascadeClassifier(cascadefile)
image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = classCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(image)
plt.show()

for i in range(len(faces)):
    print("Cadre du visage N°{0} --> {1}".format(i, faces[i]))
    plt.subplot(1, 2, i+1)
    plt.imshow(image[faces[i][1]:faces[i][1]+faces[i][3], faces[i][0]:faces[i][0]+faces[i][2]])
    plt.show()
"""
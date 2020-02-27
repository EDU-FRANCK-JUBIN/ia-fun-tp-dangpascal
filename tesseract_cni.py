try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from matplotlib import pyplot as plt

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 1)


def thresholding(image):
    return cv2.threshold(image, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


imagePath = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/IA_Fun/Ressources/CNI/ci.jpg'
dirCascadeFiles = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/IA_Fun/Ressources/OpenCv/opencv/haarcascades_cuda/'
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cascadefile = dirCascadeFiles + "haarcascade_frontalface_default.xml"
classCascade = cv2.CascadeClassifier(cascadefile)
faces = classCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
print("Il y a {0} visage(s).".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#print(pytesseract.image_to_string(image))
#plt.imshow(image)

image = cv2.imread(imagePath)
x = 151
y = 49
w = 300
h = 69
plt.imshow(cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2))

region_Nom = image[y:h, x:w]
plt.imshow(region_Nom)
plt.show()

region_Nom = remove_noise(thresholding(grayscale(region_Nom)))
NomCI = pytesseract.image_to_string(region_Nom)
print(NomCI)
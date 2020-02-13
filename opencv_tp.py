import cv2
import sys
from matplotlib import pyplot as plt

imagePath = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/image0.jpg'
dirCascadeFiles = r'/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/opencv/haarcascades_cuda'
cascadefile = dirCascadeFiles + "haarcascade_frontalface_default.xml"
classCascade = cv2.CascadeClassifier(cascadefile)
image = cv2.imread(imagePath)
plt.imshow(image)
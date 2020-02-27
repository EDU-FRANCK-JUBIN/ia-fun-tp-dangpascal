try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

#print(pytesseract.image_to_string(Image.open('/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/IA_Fun/Ressources/Tesseract/image_1.png')))
#print(pytesseract.image_to_data(Image.open('/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/IA_Fun/Ressources/Tesseract/image_1.png')))
print(pytesseract.image_to_osd(Image.open('/Users/pascaldang/Desktop/Doudou/YNOV/DeepLearning/TP/IA_Fun/Ressources/Tesseract/image_2.png')))
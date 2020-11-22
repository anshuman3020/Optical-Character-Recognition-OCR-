import cv2 
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
from sklearn import datasets, svm, metrics
from pytesseract import Output

#function to convert the image to grayscale
def gray(imgg):
    return cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

#function to blur the image
def blur(imgg):
    return cv2.GaussianBlur(imgg,(5,5),0)

#function to perform thresholding on image
def threshold(imgg):
    return cv2.adaptiveThreshold(imgg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#function to convert the image to negative
def bitwise_not(imgg):
    return cv2.bitwise_not(imgg)

#function to detect the edges of image
def edge(imgg):
    return cv2.Canny(imgg, 100, 120)

#set the size of the resized image as 400*400
IMG_SIZE = 400

#read the image 
image = cv2.imread(r'C:\Users\Apoorva\Downloads\ocr.jpg')
img2 = cv2.imread(r'C:\Users\Apoorva\Downloads\ocr.jpg')

#resize the image
img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

# call all the functions to perform image processing
gray = gray(img)
blur = blur(gray)
th = threshold(gray)
nots = bitwise_not(img)
edge = edge(img)
edge = bitwise_not(edge)

#plot the different aspects of image using subplot
#first is the gray image being plotted
plt.figure(1)
plt.subplot(221)
plt.title('gray')
plt.imshow(gray)

#second is the edged image being plotted
plt.subplot(222)
plt.title('edge')
plt.imshow(edge)

#third is the thresholded image being plotted
plt.subplot(223)
plt.title('threshold')
plt.imshow(th)

#last is the blurred image being plotted
plt.subplot(224)
plt.title('blur')
plt.imshow(blur)
plt.show()

#to detect words in the image and make boxes around them we use this function
d = pytesseract.image_to_data(edge, output_type=Output.DICT)
#the variable d now consists of a dictionary of values that the function returned

#for the number of words recognised by tesseract, we run a loop and check the
#value of each word in the conf key of d based on a certain threshold value, here: 20.
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) >-1:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        edge = cv2.rectangle(edge, (x, y), (x + w, y + h), (0, 255, 0), 2)
#for every value in conf greater that 20, tesseract will then make a rectangle
#around those words depending on the values of left, top, width and height keys
#in d

#we then display the transformed image with rectangles around the detected words
cv2.imshow('img', edge)
cv2.waitKey(0)

#this function then converts the words in the image to real time text and prints it
e = pytesseract.image_to_string(img2)
print(e)


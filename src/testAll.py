# build model and other function for neural network
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

# image processing 
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from scipy.misc import imread, imsave
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import cv2 as cv

# show the image
import matplotlib.pyplot as plt

# accessing images in directory 
import imghdr
import os 

# colored data in terminal
from termcolor import colored

# remove warnings
import warnings 


# return number of images in path directory 
def num_images(path):
    numberOfImages = 0    
    for root, dirs, files in os.walk(path):
        for name in files:
            lastPoint = name.rfind('.')     # index of last character '.'
            exstension = name[lastPoint : ]     # extension of file
            if exstension == ".png" or exstension == ".jpg" or exstension == ".jpeg":
                numberOfImages = numberOfImages + 1
    return numberOfImages


def binarize(image_path):
    img = cv.imread(image_path, 0)
    #img = cv.medianBlur(img, 5)
    #blur = cv.GaussianBlur(img,(5,3),0)
    #ret3,thresh2 = cv.threshold(blur, 0, 255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    #thresh2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,2)
    
    # make all pixels less than 128 black
    ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    
    # make all pixels less than threshold black
    imsave('binarizedImage.jpg', thresh)
    print("Binarizovana slika:")

def predictionImage():
    # remove warnins for compile model after loading
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        test_model = load_model('../model/best_model.h5')
    
    
    # open image on input path
    with open('binarizedImage.jpg', 'r') as f:
        with Image.open(f).convert('L') as image:
            
            # change size of binarized image to 28x28 pixels
            resized_image = image.resize((28,28), Image.ANTIALIAS)
            
            # plot binarized image        
            plt.imshow(resized_image)
            plt.show()
    
            # convert image to array
            x = img_to_array(resized_image)

            # add one dimension on image, [28, 28] -> [1, 28, 28]
            x = np.expand_dims(x, axis = 0)

            # get predictions for all outputs(numbers)
            predictions = test_model.predict_classes(x)
            probabilities = test_model.predict_proba(x)
            
            # write data on output
            print("Number is: " + str(predictions) + "\n")

            # remove image from disc
            os.remove('binarizedImage.jpg')


def main():
    
    # remove warnings
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # enter directory where we saved images
    directory_path = raw_input(colored("\nEnter folder path: \n", 'red'))
    directory_path = directory_path + "/"
    print(" ")
    
    # numbers of images in directory_path
    img_num = num_images(directory_path)
    print(colored('Number of images in directory "{}" is: {}\n\n'.format(directory_path, img_num), 'yellow'))
    
    # processing all image from directory
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            # define the path name
            original_image_path = directory_path + name
            print(colored('Processing image on path {}.'.format(original_image_path), 'blue'))
            
            # binarize image and predict digit
            binarize(original_image_path)
            predictionImage()
           
            
            
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
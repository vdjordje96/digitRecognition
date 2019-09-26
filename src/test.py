import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# image processing 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage
from scipy.misc import imread, imsave
import cv2 as cv

# colored data in terminal
from termcolor import colored

# remove warnings
import warnings
import os


def binarize(image_path):
    
    img = cv.imread(image_path, 0)
    #img = cv.medianBlur(img, 5)
    #blur = cv.GaussianBlur(img,(5,3),0)
    #ret3,thresh2 = cv.threshold(blur, 0, 255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # make all pixels less than 128 black
    ret, thresh = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    
    imsave('binarizedImage.jpg', thresh)

    
def predictionImage():
    # remove warnins for compile model after loading
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        test_model = load_model('../model/best_model.h5')
    
    
    # open image on input path
    with open('binarizedImage.jpg', 'r') as f:
        with Image.open(f).convert('L') as image:
            
            # change size of binarized image to 28x28 pixels
            resized_image = image.resize((28,28), Image.ANTIALIAS)
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
            print("Number is: " + str(predictions))

            # remove image from disc
            os.remove('binarizedImage.jpg')
            
            
def main():
    # remove warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
        
    # enter directory where is images
    image_path = raw_input(colored("\nEnter image path: \n", 'red'))
    print(" ")
    
    # binarize image and predict digit
    binarize(image_path)
    predictionImage()
           



if __name__ == '__main__':
    main()
    
    
    
    
    
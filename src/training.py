# for save model on Google Drive
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive 
from google.colab import auth 
from oauth2client.client import GoogleCredentials

# for define model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.optimizers import Adadelta

# for images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils.np_utils import to_categorical 

#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# image size in pixels
img_rows = 28
img_cols = 28

# each image is (1 x 28 x 28), but the Conv2D layers expect channels_last (28 x 28 x 1)
if keras.backend.image_data_format() == 'channels_first':
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape) # X_train shape: (60000, 28, 28, 1)

# set number of output categories (numbers from 0 to 9)
num_category = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category) # 2 -> [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
y_test = keras.utils.to_categorical(y_test, num_category)


# create model layer by layer
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))   
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# compile our model 
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = Adadelta(), metrics=['accuracy'])

# this metod write details of the model
model.summary()

# fast modification of images
datagen = ImageDataGenerator(
        rotation_range = 10,    # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1,       # randomly zoom image 
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1) # randomly shift images vertically (fraction of total height)

# apply generator on train data
datagen.fit(X_train)

# fitting model
batch_size = 128
num_epoch = 10
model_log = model.fit_generator(
            datagen.flow (X_train, y_train, batch_size = batch_size), 
            epochs = num_epoch,
            steps_per_epoch = X_train.shape[0] // batch_size,
            verbose = 1) 

# save the model on Google Drive
# code downloaded from: https://pythonhosted.org/PyDrive/quickstart.html
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()                       
drive = GoogleDrive(gauth)
model.save('best_model.h5')
model_file = drive.CreateFile({'title' : 'best_model.h5'})                       
model_file.SetContentFile('best_model.h5')   
model_file.Upload()

# evaluate model on test data and print result
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 

# plot the figure of accuracy and loss
plt.figure()
plt.subplot(211)
plt.plot(model_log.history['acc'])
plt.subplot(212)
plt.plot(model_log.history['loss'])
plt.show()



                                                                                                                                                                                                                                                                
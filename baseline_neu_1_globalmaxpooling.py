# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:37:24 2018

@author: PrabhakaronA
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import regularizers 
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.models import model_from_json
import sys
import cv2
import glob

#loading image using matplotlib
#img=mpimg.imread('C://Users//prabhakarona//Documents//Projects//AI//Data Sets//NEU surface defect database//Cr_1.bmp')
#Load image in grayscale using opencv
#img = cv2.imread('C://Users//prabhakarona//Documents//Projects//AI//Data Sets//NEU surface defect database//Cr_1.bmp',0)


filenames = [img for img in glob.glob("C://Users//prabhakarona//Documents//Projects//AI//Data Sets//NEU surface defect database//*.bmp")]

filenames.sort() # ADD THIS LINE

images = []
for img in filenames:
    n= cv2.imread(img, 0)
    images.append(n)
    
images = np.asarray(images)
nrows, height, width = images.shape
labels = np.zeros(shape=(1800,6))
labels[0:299, 0] = 1
labels[300:599, 1] = 1
labels[600:899, 2] = 1
labels[900:1199, 3] = 1
labels[1200:1499, 4] = 1
labels[1500:1800, 5] = 1

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.5)

#to test if y label is still correct for x after split
plt.imshow(x_train[111,:,:], cmap='gray')
print(y_train[111])

#train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)


x_train = x_train.reshape(-1, height,width, 1)
x_test = x_test.reshape(-1, height,width, 1)
x_train.shape, x_test.shape

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3)

batch_size = 32
epochs = 15
num_classes = 6
reg = 0.001

neu_model1 = Sequential()
neu_model1.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(None,None,1),padding='same', kernel_regularizer=regularizers.l2(reg)))
neu_model1.add(LeakyReLU(alpha=0.1))
#neu_model1.add(BatchNormalization())
neu_model1.add(MaxPooling2D((2, 2), padding='same'))
#neu_model1.add(Dropout(0.15))

neu_model1.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
neu_model1.add(LeakyReLU(alpha=0.1))
#neu_model1.add(BatchNormalization())
neu_model1.add(MaxPooling2D((2, 2), padding='same'))
#neu_model1.add(Dropout(0.15))

neu_model1.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
neu_model1.add(LeakyReLU(alpha=0.1))                  
#neu_model1.add(BatchNormalization())
neu_model1.add(MaxPooling2D((2, 2), padding='same'))
#neu_model1.add(Dropout(0.15))

neu_model1.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
neu_model1.add(LeakyReLU(alpha=0.1))
#neu_model1.add(BatchNormalization())                  
neu_model1.add(MaxPooling2D((2, 2), padding='same'))
neu_model1.add(Dropout(0.15))

neu_model1.add(GlobalMaxPooling2D())
neu_model1.add(Dense(128, activation='relu'))
neu_model1.add(LeakyReLU(alpha=0.1))
#neu_model1.add(BatchNormalization())
neu_model1.add(Dropout(0.5))
neu_model1.add(Dense(num_classes, activation='softmax'))

neu_model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr = 0.0005), metrics=['accuracy'])
#neu_model1 = load_model('C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\relu_b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax_.h5')
model_train = neu_model1.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, y_valid))
neu_model1.summary()
from keras.preprocessing import image
test_image = image.load_img("C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\test_10.bmp", grayscale = True)
#test_image = np.asarray(x_test[14])

test_image = np.asarray(test_image)
print (test_image.shape)
nrows, height, width = images.shape
test_image = test_image.reshape(-1, height, width, 1)
test_image = test_image.astype('float32')
test_image = test_image / 255.
result = neu_model1.predict(test_image)
if result[0].argmax() == 0:
    prediction = 'Crazing'
elif result[0].argmax() == 1:
    prediction = 'Inclusion'    
elif result[0].argmax() == 2:
    prediction = 'Pitted Surface'
elif result[0].argmax() == 3:
    prediction = 'Patches'
elif result[0].argmax() == 4:
    prediction = 'Rolled-in-scale'
elif result[0].argmax() == 5:
    prediction = 'Scratches'

print (result)
print ("\n The defect is", prediction)
#np.testing.assert_allclose(neu_model2.predict(x_test), neu_model1.predict(x_test), 1e-5)
#epochs = 100
#neu_model_train = neu_model1.fit_generator(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, y_valid))
#keras.callbacks.CSVLogger('C:\\Users\\prabhakarona\\Documents\\Digital General\\Learning\\Computer Vision\\Problem Sets and codes\\CNN DataCamp Example\\test.csv', separator=',', append=True)

test_pred = neu_model1.predict(x_test, verbose = 1)
test_eval = neu_model1.evaluate(x_test, y_test, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#y_pred = np.zeros(len(y_test))
#y_pred.astype(int)
#for i in test_pred:
#    y_pred[i] = np.amax(test_pred)

####Convert predicted and actual test labels from categorical to values (ie 0 to 5) to compute Confusion Matrix

y_test1 = [ np.argmax(t) for t in y_test]
y_pred = [ np.argmax(t) for t in test_pred]

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test1, y_pred)
print ('Confusion matrix: ', conf_matrix)

accuracy = np.array(model_train.history['acc'])
val_accuracy = np.array(model_train.history['val_acc'])
loss = np.array(model_train.history['loss'])
val_loss = np.array(model_train.history['val_loss'])
test = np.array(test_eval)
test_accuracy = np.array(test[1])
test_loss = np.array(test[0])

#np.savetxt("C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\confusion_matrix__b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax__random8.csv", conf_matrix,  delimiter=",", fmt = '%.4f', newline='\n')
#np.savetxt("C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\relu_b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax__random8.csv", np.transpose([accuracy, val_accuracy, loss, val_loss]),  delimiter=",", fmt = '%.4f', header = "Training Accuracy, Validation_Acc, Training Loss, Validation_Loss", newline='\n', comments = "")
#np.savetxt("C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\test_b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax__random8.csv",np.transpose([test[1], test[0]]),  delimiter=",", fmt = '%.4f', header = "test_Acc, test_loss", newline='\n', comments = "")

plt.figure(1)  

########## Visualization DC of results to detect overfitting ##########
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('No of epochs')
plt.ylabel('Accuracy/Loss Value')
plt.savefig('C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\relu_b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax__random8_Accuracy.png', bbox_inches='tight')

plt.figure(2)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.xlabel('No of epochs')
plt.ylabel('Accuracy/Loss Value')
#plt.show()  
#plt.savefig('C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\relu_b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax__random8_Loss.png', bbox_inches='tight')
#neu_model1.save('C:\\Users\\prabhakarona\\Documents\\Projects\\AI\\Data Sets\\Results\\NEU_Baseline\\relu_b32_e15_conv32_64_128_256_drop256-15_50_l2reg0.001_lr0.0005_globalmax__random8.h5')

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#with device('/cpu:0'):

#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))



# Deep Learning CNN model to recognize face
'''This script uses a database of images and creates CNN model on top of it to test
   if the given image is recognized correctly or not'''

'''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

# Specifying the folder where images are present
TrainingImagePath='./images/train'
TestingImagePath='./images/train'

from keras.preprocessing.image import ImageDataGenerator
# Understand more about ImageDataGenerator at below link
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Defining pre-processing transformations on raw images of training data
# These hyper parameters helps to generate slightly twisted versions
# of the original image, which leads to a better model, since it learns
# on the good and bad mix of images
train_datagen = ImageDataGenerator(
		shear_range=0.1,
		rotation_range=50,
		brightness_range=[0.1,2.5],
		zoom_range=0.1,
		horizontal_flip=True)

# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()

# Generating the Training Data
training_set = train_datagen.flow_from_directory(
		TrainingImagePath,
		target_size=(64, 64),
		batch_size=32,
		class_mode='categorical')

len(training_set)

# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
		TestingImagePath,
		target_size=(64, 64),
		batch_size=32,
		class_mode='categorical')

# Printing class labels for each face
test_set.class_indices

len(training_set)

'''############ Creating lookup table for all faces ############'''
# class_indices have the numeric tag for each face
TrainClasses=training_set.class_indices

# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
	ResultMap[faceValue]=faceName

# Saving the face map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
	pickle.dump(ResultMap, fileWriteStream)

# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("Mapping of Face and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)






'''######################## Create CNN deep learning model ########################'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

'''Initializing the Convolutional Neural Network'''
classifier= Sequential()
 
''' STEP--1 Convolution
# Adding the first layer of CNN
# we are using the format (64,64,3) because we are using TensorFlow backend
# It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
'''
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
 
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
 
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
 
 
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.5)) 

classifier.add(Dense(OutputNeurons, activation='softmax'))
 
'''# Compiling the CNN'''
#classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
 
###########################################################
import time
# Measuring the time taken by the model to train
StartTime=time.time()
 
# Starting the model training
classifier.fit_generator(
					training_set,
					steps_per_epoch=5,
					epochs=300,
					validation_data=test_set,
					validation_steps=1)
 
EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')

classifier.save_weights('./model/face_detect.h5')




'''########### Making single predictions ###########'''
import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
import os.path as p
import os

def predict(impath):
		if not p.isfile(impath):
				return False
		test_image=load_img(impath,target_size=(64, 64))
		test_image=img_to_array(test_image)
		test_image=np.expand_dims(test_image,axis=0)
		result=classifier.predict(test_image,verbose=0)
		print('####'*10)
		print(impath)
		print('Prediction is: ',ResultMap[np.argmax(result)])
		return ResultMap[np.argmax(result)]

import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

predict('./images/wild/0.png')
import os

testfolder = './images/test/others/'
total =0
mistake = 0
for file in os.listdir(testfolder):
	total +=1
	res = predict(testfolder+file)
	if  res == 'frames' and file.startswith('frame'):
		print('successful identified frame class for ' + testfolder+file)
	elif res == 'others' and  file.startswith('frame'):
		print('UNSUCCESSFUL for frame class: resulted others ' + testfolder + file)
		mistake +=1
	elif res == 'frames' and not file.startswith('frame'):
		print('UNSUCCESSFULL FOR ' + testfolder + file)
		mistake += 1
	else: print('successful others for ' + testfolder + file)

	print('-'*3)

print((total -mistake)/total)

classifier.load_weights('./model/face_detect.h5')


TrainingImagePath='./ms'
TestingImagePath='./ms'

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

len(training_set)

test_set = test_datagen.flow_from_directory(
        TestingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set.class_indices

len(training_set)

TrainClasses=training_set.class_indices

ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

print("Mapping of Face and its ID",ResultMap)

OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
classifier= Sequential()
 
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu')) 
classifier.add(MaxPool2D(pool_size=(2,2)))


classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
 
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
# classifier.add(MaxPool2D(pool_size=(2,2)))

 
classifier.add(Flatten())
 
classifier.add(Dense(64, activation='relu'))
 
classifier.add(Dense(OutputNeurons, activation='relu'))
 
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
 
import time
StartTime=time.time()
 
classifier.fit_generator(
                    training_set,
                    steps_per_epoch=1,
                    epochs=1000,
                    validation_data=test_set,
                    validation_steps=1)
 
EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')
classifier.save_weights('./model/face_detect.h5')
import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
import os.path as p


def predict(impath):
	if not p.isfile(impath):
		return False
	test_image=load_img(impath,target_size=(64, 64))
	test_image=img_to_array(test_image)
	test_image=np.expand_dims(test_image,axis=0)
	result=classifier.predict(test_image,verbose=0)
	print('####'*10)
	print('Prediction is: ',ResultMap[np.argmax(result)])
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

screen_width = 1280
screen_height = 720

stream = cv2.VideoCapture(0)

frame_num = 0
while(True):
    (pic, frame) = stream.read()
    if not pic:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)
    face = 0
    for (x, y, w, h) in faces:
        face+=1
        color = (0,255,255)
        stroke = 5;
        name = "C:\\general\\ai2\\temp\\tmp.jpg"
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        result = predict('./temp/tmp.jpg')
        print(ResultMap[np.argmax(result)])
    cv2.imshow('image', frame)
    key = cv2.waitKey(1) & 0xFF
    frame_num+=1
    if key == ord('q'): break

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

predict('./mstest/27.png')
predict('./mstest/41.png')
predict('./mstest/63.png')
predict('./mstest/65.png')
predict('./mstest/68.png')
predict('./mstest/70.png')
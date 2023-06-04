
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
TrainingImagePath='./images/train'
TestingImagePath='./images/test'



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
		target_size=(150,150),
		batch_size=32,
		shuffle=True,
		class_mode='categorical')

len(training_set)

# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
		TestingImagePath,
		target_size=(150, 150),
		batch_size=32,
		shuffle=True,
		class_mode='categorical')

# Printing class labels for each face
test_set.class_indices

len(training_set)

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


from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, MaxPooling2D, BatchNormalization
import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
import os.path as p
import time

from keras.preprocessing.image import ImageDataGenerator



def create_model():
    model = Sequential([
        Conv2D(filters=128, kernel_size=(5, 5), padding='valid', input_shape=(150, 150, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(filters=64, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(0.00005)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(filters=32, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(0.00005)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(units=256, activation='relu'),
        Dropout(0.5),
        Dense(units=4, activation='softmax')
    ])
    
    return model

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
cnn = create_model()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5)

optimizer = Adam(learning_rate=0.001)
cnn.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])


history = cnn.fit(training_set, epochs=50, validation_data=test_set,
                       verbose=2,
                       callbacks=[reduce_lr])


import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
import os.path as p
import os

def predict(impath):
		if not p.isfile(impath):
				return False
		test_image=load_img(impath,target_size=(150, 150))
		test_image=img_to_array(test_image)
		test_image=np.expand_dims(test_image,axis=0)
		result=cnn.predict(test_image,verbose=0)
		print('####'*10)
		print(impath)
		print('Prediction is: ',ResultMap[np.argmax(result)])
		return ResultMap[np.argmax(result)]


testfolder = './images/test/others/'
total =0
mistake = 0
for file in os.listdir(testfolder):
	total +=1
	res = predict(testfolder+file)
	print(res)

	print('-'*3)

predict('./temp/tmp.jpg')

import cv2

def cap():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    screen_width = 1280
    screen_height = 720
    stream = cv2.VideoCapture(0)
    mistakes = 0
    frame_num = 0
    while(True):
        (pic, frame) = stream.read()
        if not pic:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)
        face = 0
        faces

        for (x, y, w, h) in faces:
            face+=1
            color = (0,255,255)
            stroke = 5;
            name = "C:\\general\\ai2\\temp\\tmp.jpg"
            cv2.imwrite(name, frame[y:y+h, x:x+w])
            result = predict('./temp/tmp.jpg')
            cv2.putText(frame, result, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=1, color=0)
            print(result)
            print()
            # if(result != 'Nishedh'):
            #     mistakes +=1
            #     cv2.imwrite('C:\\general\\ai2\\images\\train\\Nishedh\\nishedh_new_%d.jpg'%face, frame[y:y+h, x:x+w])

            frame_num+=1
            cv2.imshow('image', frame)
            key = cv2.waitKey(500) & 0xFF
            if key == ord('q'): break

        

    stream.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f'{mistakes=}, {frame_num=}, accuracy={(frame_num -mistakes)/frame_num}')

cap()


(x,y,z) =()
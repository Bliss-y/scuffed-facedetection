
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
import os.path as p
import time

from keras.preprocessing.image import ImageDataGenerator


class model:
    def __init__(self, OutputNeurons=6) -> None:
        self.classifier= Sequential();
        self.classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(2,2)))

        self.classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(2,2)))

        self.classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(2,2)))

        self.classifier.add(Flatten())
        self.classifier.add(Dense(64, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(OutputNeurons, activation='softmax'))
        self.classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

    def load(self):
        self.classifier.load_weights('./model/face_detect.h5')    

    


    def predict(self, impath):
        if not p.isfile(impath):
            return False
        test_image=load_img(impath,target_size=(64, 64))
        test_image=img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=self.classifier.predict(test_image,verbose=0)
        print('####'*10)
        print('Prediction is: ',self.ResultMap[np.argmax(result)])
        return self.ResultMap[np.argmax(result)]

    def train(self, epochs=500):
        StartTime=time.time()
 
        self.classifier.fit_generator(
                            self.training_set,
                            steps_per_epoch=3,
                            epochs=epochs,
                            validation_data=self.test_set,
                            validation_steps=3)
        
        EndTime=time.time()
        print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')
        self.classifier.save_weights('./model/face_detect.h5')


    def createDataSet(TrainingImagePath, TestingImagePath):
        train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

        test_datagen = ImageDataGenerator()

        training_set = train_datagen.flow_from_directory(
                TrainingImagePath,
                target_size=(64, 64),
                batch_size=2,
                class_mode='categorical')

        test_set = test_datagen.flow_from_directory(
                TestingImagePath,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')
        
        
        TrainClasses=training_set.class_indices
        ResultMap={}
        for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
            ResultMap[faceValue]=faceName
        import pickle
        with open("ResultsMap.pkl", 'wb') as fileWriteStream:
            pickle.dump(ResultMap, fileWriteStream)
        OutputNeurons=len(ResultMap)
        m = model(OutputNeurons=OutputNeurons)
        m.training_set = training_set
        m.ResultMap = ResultMap
        m.test_set = test_set
        return m

model.createDataSet = staticmethod(model.createDataSet)
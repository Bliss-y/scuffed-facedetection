
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
from os import path as p, mkdir, listdir
import time
from keras.preprocessing.image import ImageDataGenerator
import json


class model:
    def __init__(self, OutputNeurons=6, dropout = True) -> None:
        self.arch= Sequential();
        self.arch.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
        self.arch.add(MaxPool2D(pool_size=(2,2)))

        self.arch.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.arch.add(MaxPool2D(pool_size=(2,2)))

        self.arch.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.arch.add(MaxPool2D(pool_size=(2,2)))

        self.arch.add(Flatten())
        self.arch.add(Dense(64, activation='relu'))
        if dropout: self.arch.add(Dropout(0.5))
        self.arch.add(Dense(OutputNeurons, activation='softmax'))
        self.arch.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

    def load(dir, name):
        f = open(dir+name+'/model.json', 'r')
        ResultMap = json.load(f)
        c = {}
        it = 0
        for i in ResultMap:
            c[it] = ResultMap[str(it)];
            it+=1
        f.close()
        ResultMap = c
        OutputNeurons = len(ResultMap)
        m = model(OutputNeurons=OutputNeurons)
        m.ResultMap = ResultMap
        m.OutputNeurons = OutputNeurons
        m.arch.load_weights(dir+name + '/model.h5')
        return m
        

    def saveModel(self, dir, name):
        if not p.exists(dir + name):
            mkdir(dir + name)
        f = open(dir + name + '/model.json', 'w')
        self.arch.save_weights(p.join( dir+name + '/model.h5'))
        json.dump(self.ResultMap, f)
        f.close()
        self.OutputNeurons = len(self.ResultMap)

    def predict(self, impath):
        if not p.isfile(impath):
            return False
        test_image=load_img(impath,target_size=(64, 64))
        test_image=img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=self.arch.predict(test_image,verbose=0)
        return self.ResultMap[np.argmax(result)]
    
    def predictFolder(self, testfolder):
        total =0
        for file in listdir(testfolder):
            total +=1
            res = self.predict(testfolder+file)
            print(testfolder + file)
            print(res)
            print('-'*3)

    def train(self, epochs=500):
        StartTime=time.time()
 
        self.arch.fit_generator(
                            self.training_set,
                            steps_per_epoch=27,
                            epochs=epochs,
                            validation_data=self.test_set,
                            validation_steps=10)
        
        EndTime=time.time()
        print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')


    def createDataSet(TrainingImagePath, TestingImagePath):
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
model.load = staticmethod(model.load)
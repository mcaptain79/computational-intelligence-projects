"""
Created on Fri Mar 26 15:35:03 2021

@author: mcaptain79
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#function below is for reading mnist data set
def read_data():
    (trainData,trainLabels),(testData,testLabels) = keras.datasets.mnist.load_data()
    return (trainData/256,trainLabels),(testData/256,testLabels)
#reading these data from function
(trainData,trainLabels),(testData,testLabels) = read_data()
#function below is for showing image and its label for train set
def show_train_image(number):
    print('label:',trainLabels[number])
    plt.imshow(trainData[number])
    plt.show()
#function below is for showing image and its label for test set
def show_test_image(number):
    print('label:',testLabels[number])
    plt.imshow(testData[number])
    plt.show()
#showing number of data for train and test
def dataNumber():
    print('train::',len(trainData))
    print('test:',len(testData))
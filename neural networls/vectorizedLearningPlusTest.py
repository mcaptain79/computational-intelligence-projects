"""
Created on Mon Mar 29 01:13:19 2021

@author: mcptain79
"""
from readData import *
import numpy as np
import math
import matplotlib.pyplot as plt
"""
     fourth step of the project
"""
#reading our data first
(trainData,trainLabels),(testData,testLabels) = read_data()
#function below is implementation of relu function and adding it to numpy library
def activation_function(x):
    if x > 0:
        return x
    return 0
def activation_function_prime(x):
    if x > 0:
        return 1
    else:
        return 0
activation_function = np.frompyfunc(activation_function,1,1)
activation_function_prime = np.frompyfunc(activation_function_prime,1,1)
#find maximum element position in a numpy array
def find_max_element(matrix,row,column):
    resRow = 0
    resCol = 0
    maximum= matrix[0,0]
    for i in range(row):
        for j in range(column):
            if maximum < matrix[i][j]:
                resRow = i
                resCol = j
                maximum = matrix[i][j]
    return (resRow,resCol)
#we should initialize weights and bias for each level
#weights for each level
weights_level1 = np.random.normal(size = (16,784))
weights_level2 = np.random.normal(size = (16,16))
weights_level3 = np.random.normal(size = (10,16))
#bias for each level
bias_level1 = np.zeros((16,1))
bias_level2 = np.zeros((16,1))
bias_level3 = np.zeros((10,1))
#calcuting cost function
def cost(resLabels,k):
    trueLabels = [0]*10
    trueLabels[trainLabels[k]] = 1
    res = 0
    for i in range(10):
        res += math.pow(trueLabels[i]-resLabels[i],2)
    return res
#function below is for calculating a and z s 
#second level for ith element
def level2_AZ(i):
    z2 = np.dot(weights_level1,trainData[i].reshape(784,1))+bias_level1
    return (z2,activation_function(z2))
#third level and passing a2 as argument
def level3_AZ(a2):
    z3 = np.dot(weights_level2,a2)+bias_level2
    return (z3,activation_function(z3))
#function below is for calculating the result
def res_calculator(i):
    z2,a2 = level2_AZ(i)
    z3,a3 = level3_AZ(a2)
    res = np.dot(weights_level3,a3)+bias_level3
    return (res,activation_function(res))
#function below is for calculating real result labels
def y_calculator(i):
    y = np.zeros((10,1))
    y[trainLabels[i],0] = 1
    return y
#algortithm below is for learning
def learn(learning_rate,number_of_epoches,batch_size):
    global weights_level3,weights_level2,weights_level1,bias_level1,bias_level2,bias_level3
    for i in range(number_of_epoches):
        #we should initialize weights and bias gradients
        #weight gradients
        weights_level1_gradient = np.zeros((16,784))
        weights_level2_gradient = np.zeros((16,16))
        weights_level3_gradient = np.zeros((10,16))
        #bias gradients
        bias_level1_gradient = np.zeros((16,1))
        bias_level2_gradient = np.zeros((16,1))
        bias_level3_gradient = np.zeros((10,1))
        #a2 and a3 gradient
        a2_gradient = np.zeros((16,1))
        a3_gradient = np.zeros((16,1))
        for j in range(batch_size):
            for k in range(j*6000,j*6000+6000):
                z2,a2 = level2_AZ(k)
                z3,a3 = level3_AZ(a2)
                res = np.dot(weights_level3,a3)+bias_level3
                y = y_calculator(k)
                calc = 2*activation_function_prime(res)*(activation_function(res)-y)
                a2_gradient = a2_gradient+np.dot(weights_level3.transpose(),calc)
                weights_level3_gradient = weights_level3_gradient+np.dot(calc,a3.transpose())
                middle21 = np.dot(weights_level3.transpose(),calc)
                weights_level2_gradient = weights_level2_gradient + np.dot(middle21,(activation_function_prime(z3)*a2).transpose())
                middle11 = np.dot(weights_level3.transpose(),calc)
                middle12 = np.dot(middle11,activation_function_prime(z3).transpose())
                middle13 = np.dot(np.dot(weights_level2,activation_function_prime(z2)),trainData[k].reshape(1,784))
                weights_level1_gradient = weights_level1_gradient + np.dot(middle12,middle13)
            weights_level3 = weights_level3 - (learning_rate*(weights_level3_gradient/batch_size))
            weights_level2 = weights_level2 - (learning_rate*(weights_level2_gradient/batch_size))
            weights_level1 = weights_level1 - (learning_rate*(weights_level1_gradient/batch_size))
"""
    fifth step of project
"""
def test(testData,testLabels):
    counter = 0
    learn(1, 20, 10)
    for i in range(len(testData)):
        res = res_calculator(i)[0]
        if find_max_element(res, 10, 1)[0] == trainLabels[i]:
            counter += 1
    return counter/100
print('accuracy:','%'+str(test(testData, testLabels)))



















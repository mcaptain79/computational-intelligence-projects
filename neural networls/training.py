"""
Created on Fri Mar 26 16:01:28 2021

@author: mcaptain79
"""
import numpy as np
import matplotlib.pyplot as plt
from readData import *
import math
#reading mnist data form our written function
(trainData,trainLabels),(testData,testLabels) = read_data()
"""
      the second step
"""
#creating random weights
weights = np.random.normal(size = (10,784))
#function below is implementation of sigmoid function
def sigmoid(x):
    return 1/(1+math.pow(math.e,-1*x))
#adding sigmoid function to numpy library
sigmoid = np.frompyfunc(sigmoid, 1, 1)
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
#making bias
bias = np.zeros((10,1))
#using first hundred pictures
resLabelList = []
for i in range(100):
    resMatrix = np.dot(weights,trainData[i].reshape(784,1)) + bias
    resLabelList.append(find_max_element(resMatrix, 10, 1)[0])
#calculating the accuracy
hits = 0
for i in range(100):
    if resLabelList[i] == trainLabels[i]:
        hits += 1
print('accuracy for second step is: %'+str(hits))
"""
    the third step
"""
#these two matrix below are for gradian and we initializes their members to zero
weight_gradient = np.zeros((10,784))
bias_gradient = np.zeros((10,1))
#functions below are for calculating dcost/dweight and dcost/dbias cost gradind for both
"""
weight and bias matrix are our base matrix
dataNum is number of the wanted train data
and function below computes gradient just for one single picture
"""
def cost_gradient_calc(weightMatrix,biasMatrix,dataNum):
    dcost_dweight = np.zeros((10,784))
    dcost_dbias = np.zeros((10,1))
    z = [0]*10
    for i in range(10):
        z[i] += np.dot(weightMatrix[i],trainData[dataNum].reshape(784,1))
    a = [sigmoid(x) for x in z]
    y = [0]*10
    y[trainLabels[dataNum]] = 1
    for i in range(10):
        for j in range(784):
            dcost_dweight[i,j] = 2*(a[i]-y[i])*sigmoid(z[i])*(1-sigmoid(z[i]))*trainData[dataNum].reshape(784,1)[j]
    for i in range(10):
        dcost_dbias[i,0] = 2*(a[i]-y[i])*sigmoid(z[i])*(1-sigmoid(z[i]))
    return (dcost_dweight,dcost_dbias)
#these are our hyper parameters and we should initialize it
learning_rate = 1
number_of_epoches = 20
batch_size = 10
#list below is for average of costs
cost_avgs = []
#algorithm below is for learning
for i in range(number_of_epoches):
    for j in range(batch_size):
        weight_gradient = np.zeros((10,784))
        bias_gradient = np.zeros((10,1))
        for k in range(j*10,j*10+10):
            dcost_dweight,dcost_dbias = cost_gradient_calc(weights,bias,k)
            weight_gradient += dcost_dweight
            bias_gradient += dcost_dbias
        weights = weights - (learning_rate*(weight_gradient/batch_size))
        bias = bias - (learning_rate*(bias_gradient/batch_size))
    resLabelList3 = []
    for i in range(100):
        c_list = []
        resMatrix = np.dot(weights,trainData[i].reshape(784,1)) + bias
        resLabelList3.append(find_max_element(resMatrix, 10, 1)[0])
        c_list.append(math.pow(resLabelList3[i]-trainLabels[i],2))
    cost_avgs.append(sum(c_list)/100)
#plotting cost function
plt.plot(cost_avgs)
plt.show()            
#using first hundred pictures
resLabelList2 = []
for i in range(100):
    resMatrix2 = np.dot(weights,trainData[i].reshape(784,1)) + bias
    resLabelList2.append(find_max_element(resMatrix2, 10, 1)[0])
#calculating the accuracy
hits2 = 0
for i in range(100):
    if resLabelList2[i] == trainLabels[i]:
        hits2 += 1
print('accuracy for third step is: %'+str(hits2))
            
            














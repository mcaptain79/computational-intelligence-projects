import numpy as np
import math

class NeuralNetwork():

    def __init__(self, layer_sizes):

        # layer_sizes example: [4, 10, 2]
        self.layer_sizes = layer_sizes
        self.w1 = np.random.normal(size = (self.layer_sizes[1],self.layer_sizes[0]))
        self.w2 = np.random.normal(size = (self.layer_sizes[2],self.layer_sizes[1]))
        self.b1 = np.zeros((self.layer_sizes[1],1))
        self.b2 = np.zeros((self.layer_sizes[2],1))

    def activation(self, x):
        
        return 1/(1+math.pow(math.e,-1*x))

    def forward(self, x):
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        activation2 = np.frompyfunc(self.activation,1,1)
        l2 = activation2(self.w1 @ x + self.b1)
        l3 = activation2(self.w2 @ l2 + self.b2)
        return l3
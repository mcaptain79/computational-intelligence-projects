from player import Player
import numpy as np
from config import CONFIG
import random
from nn import *
import copy
class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        w1Rows,w1Columns = child.nn.w1.shape
        w2Rows,w2Columns = child.nn.w2.shape
        b1Rows,b1Columns = child.nn.b1.shape
        b2Rows,b2Columns = child.nn.b2.shape
        for i in range(w1Rows):
            for j in range(w1Columns):
                randomNum = random.randrange(1,11)
                if randomNum == 1 or randomNum == 2:
                     child.nn.w1[i,j] += random.randrange(-10000,10000)/10000
        for i in range(w2Rows):
            for j in range(w2Columns):
                randomNum = random.randrange(1,11)
                if randomNum == 1 or randomNum == 2:
                     child.nn.w2[i,j] += random.randrange(-10000,10000)/10000
        for i in range(b1Rows):
            for j in range(b1Columns):
                randomNum = random.randrange(1,11)
                if randomNum == 1 or randomNum == 2:
                     child.nn.b1[i,j] += random.randrange(-10000,10000)/10000
        for i in range(b2Rows):
            for j in range(b2Columns):
                randomNum = random.randrange(1,100)
                if randomNum == 1:
                     child.nn.b2[i,j] += random.randrange(-10000,10000)/10000
        return child
    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover
            res = []
            myWeights = []
            for i in range(num_players):
                myWeights.append(prev_players[i].fitness)
            for i in range(num_players):
                x = random.choices(prev_players,weights = myWeights)
                param = copy.deepcopy(x[0])
                res.append(self.mutate(param)) 
            return res

    def next_population_selection(self, players, num_players):

        # num_players example: 100
        # players: an array of `Player` objects
        players = sorted(players)
        res = []
        for i in range(num_players):
            res.append(max(random.choices(players,k = 5)))
        return res

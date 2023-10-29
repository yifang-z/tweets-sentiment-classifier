import time
import numpy as np
from collections import defaultdict

class maxEnt:
    def __init__(self, trainDataList, trainLabelList):
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.featureNum = len(trainDataList[0])
        self.N = len(trainDataList)
        self.n = 0
        self.M = 10000
        self.fixy = self.calc_time()
        self.w = [0] * self.n
        self.xy2idDict, self.id2xyDict = self.createdict()
        self.Ep_xy = self.calc_expectance()

    def calcep_empirical(self):
        Epxy = [0] * self.n
        for i in range(self.N):
            Pwxy = [0] * 2
            Pwxy[0] = self.calcPwy_x(self.trainDataList[i], 0)
            Pwxy[1] = self.calcPwy_x(self.trainDataList[i], 1)
            for feature in range(self.featureNum):
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
        return Epxy

    def calc_expectance(self):
        Ep_xy = [0] * self.n
        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                id = self.xy2idDict[feature][(x, y)]
                Ep_xy[id] = self.fixy[feature][(x, y)] / self.N
        return Ep_xy

    def createdict(self):
        xy2idDict = [{} for i in range(self.featureNum)]
        id2xyDict = {}
        index = 0
        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                xy2idDict[feature][(x, y)] = index
                id2xyDict[index] = (x, y)
                index += 1
        return xy2idDict, id2xyDict


    def calc_time(self):
        fixyDict = [defaultdict(int) for i in range(self.featureNum)]
        for i in range(len(self.trainDataList)):
            for j in range(self.featureNum):
                fixyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] += 1
        for i in fixyDict:
            self.n += len(i)
        return fixyDict


    def calcPwy_x(self, X, y):
        numerator = 0
        Z = 0
        for i in range(self.featureNum):
            if (X[i], y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], y)]
                numerator += self.w[index]
            if (X[i], 1-y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], 1-y)]
                Z += self.w[index]
        numerator = np.exp(numerator)
        Z = np.exp(Z) + numerator
        return numerator / Z


    def maxEntropyTrain(self, iter = 50):
        for i in range(iter):
            iterStart = time.time()
            Epxy = self.calcep_empirical()
            sigmaList = [0] * self.n
            for j in range(self.n):
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / (Epxy[j]+1e-5))
            self.w = [self.w[i] + sigmaList[i] for i in range(self.n)]
            iterEnd = time.time()
            print('iter:%d:%d, time:%d'%(i, iter, iterStart - iterEnd))

    def predict(self, X):
        result = [0] * 2
        for i in range(2):
            result[i] = self.calcPwy_x(X, i)
        return result.index(max(result))

    def test(self, testDataList):
        result = []
        for i in range(len(testDataList)):
            result.append(self.predict(testDataList[i]))
        return result

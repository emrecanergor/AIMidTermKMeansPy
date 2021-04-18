import random
import math
import copy

class KMeansAlgorithm():
    def __init__(self, dataset, K, threshold):
        self.dataset = dataset
        self.K = K
        self.numDim, self.dataSize = self._checkDim()
        self.MinMax={}
        self.centroid={}
        self._getMinMax()
        self._randomCentroid()
        self.classes=[]
        self.threshold = threshold

    def _checkDim(self):
        numDim = 0
        dataSize = len(self.dataset)
        for i in range(len(self.dataset[0])):
            numDim+=1
        return numDim, dataSize

    def _getMinMax(self):
        for dim in range(self.numDim):
            maxValue = self.dataset[0][dim]
            minValue = self.dataset[0][dim]
            for num in range(self.dataSize):
                maxValue = max(maxValue, self.dataset[num][dim])
                minValue = min(minValue, self.dataset[num][dim])
            self.MinMax[dim]=[minValue, maxValue]

    def _randomCentroid(self): 
        for k in range(self.K):
            random_centroid=[]
            for dim in range(self.numDim):
                random_centroid.append(random.uniform(self.MinMax[dim][0], self.MinMax[dim][1]))
            self.centroid[k]= random_centroid

    def _euclideanDistance(self, datum): 
        euclideanResult = []
        for k in range(self.K):
            euclideanValue = 0
            for dim in range(self.numDim):
                value = self.centroid[k][dim] - datum[dim]
                euclideanValue += value**2
            euclideanResult.append(math.sqrt(euclideanValue))
        euclideanResult = self._oneHotEncoding(euclideanResult)
        return euclideanResult

    def fit(self):
        changeInPosition = self.threshold+1 
        numIteration = 0
        while(self.threshold<changeInPosition):
            numIteration+=1
            result=[]
            for i in range(len(self.dataset)):
                result.append(self._euclideanDistance(self.dataset[i]))
            self.centroid_old = copy.deepcopy(self.centroid)
            num = {}
            for i in range(self.K):
                self.centroid[i] = [0]*self.numDim
                num[i] = 0

            for i, datum in zip(result, range(len(self.dataset))):
                for z, classNum in zip(i, range(self.K)):
                    if z == 1:
                        num[classNum]+=1
                        self.centroid[classNum] = [x + y for x, y in zip(self.centroid[classNum], self.dataset[datum])]
            RMSE = 0
            for i in range(self.K):
                if(num[i] != 0):
                    self.centroid[i] = [value/num[i] for value in self.centroid[i]]
                else:
                    self.centroid[i] = [value for value in self.centroid[i]]
                RMSE = sum([math.sqrt(abs(x**2-y**2)) for x,y in zip(self.centroid_old[i], self.centroid[i])])
            changeInPosition = RMSE
            print("Iterasyon Sayisi "+str(numIteration)+" : "+str(changeInPosition)+" <- Mesafe Degisimi")
            self.classes = result

    def _oneHotEncoding(self, data):
        result = data
        m = min(data)
        result = [1 if r==m else 0 for r in result]
        return result
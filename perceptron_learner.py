from __future__ import (absolute_import, division, print_function, unicode_literals)
from .supervised_learner import SupervisedLearner
import random

class PerceptronLearner(SupervisedLearner):
    inputs,targets,weights,bestWeights = [],[],[],[]
    trainingepoch,unprogressiveEpochs,correctInEpoch,epochLimit = 0,0,0,3
    learningrate=.2
    currentAccuracy,bestAccuracy = 0.0,0.0

    def train(self, inputs_matrix, targets_matrix):
      self.initData(inputs_matrix,targets_matrix)
      while self.currentAccuracy!=1.0 and self.unprogressiveEpochs<self.epochLimit:
        self.currentAccuracy=self.runEpoch()
        if self.currentAccuracy>self.bestAccuracy:
          self.bestAccuracy=self.currentAccuracy
          self.bestWeights=[weight for weight in self.weights]
        else:
          self.unprogressiveEpochs+=1
      print("Total epochs run: "+str(self.trainingepoch))
      self.weights=self.bestWeights

    def runEpoch(self):
      self.trainingepoch+=1
      self.correctInEpoch=0
      for idx,vector in enumerate(self.inputs):
        output=self.checkVector(vector)
        if output!=int(self.targets[idx][0]):
          self.changeWeights(self.targets[idx][0],output,vector)
        else:
          self.correctInEpoch+=1
      return self.correctInEpoch/len(self.inputs)

    def predict(self, features, targets):
      if len(features)<len(self.weights): # we're now predicting new data, so we should also add in a bias node
        features.append(1)
      targets += [self.checkVector(features)]

    def initData(self,inputs_matrix,targets_matrix):
      self.inputs=self.addBias([inputvector for inputvector in inputs_matrix.data])
      self.targets=[target for target in targets_matrix.data]
      self.weights=[round(random.uniform(-1,1),1) for val in self.inputs[0]]

    def addBias(self,vectors):
      for vector in vectors:
        vector.append(1) # add bias node to each input vector
      return vectors

    def changeWeights(self,target,output,inputvector):
      change=[self.learningrate*(int(target)-output)*inputvector[idx] for idx in range(len(self.weights))]
      self.weights=[round(self.weights[i]+change[i],2) for i in range(len(self.weights))]

    def checkVector(self,vector):
      activation=0
      for k,v in enumerate(vector):
        activation += self.weights[k]*v
      return 1 if activation > 0 else 0


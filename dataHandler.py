'''
  Justin Chen
  9.29.16
  dataHandler.py

  This module generates the data.

  Boston University Small Brain Project
  Policy Smoothing with Neural Networks
'''
import random
import numpy as np

class DataHandler():

	def __init__(self, dataParams=[], batchSize=1):
		#assuming the input formmated as a 2-tuple where first element is the feature vectors 
		# and the second element is the output layer size (number of expected outputs e.g. 1 for regression and >= for classification)
		#User should also pass in a labeling function that calculates the output given a sample
		self.features  = dataParams[0]
		self.outSize   = dataParams[1]
		self.batchSize = batchSize
	
	def __label(self, sample):
		label = np.empty((1, self.outSize))
		np.append(label, np.sum(sample)) #np.array([np.sum(sample)]) #maybe be b/c the np.sum make a new ndarray and then nesting it in the empty ndarray
		return label

	def sample(self):
		sample = np.empty((1, len(self.features)))
		for f in self.features:
			np.append(sample, random.uniform(f[0], f[1]))
		return sample

	def getBatch(self):
		features = np.empty((self.batchSize, len(self.features)))
		labels	 = np.empty((self.batchSize, self.outSize))

		for b in range(self.batchSize):
			sample = self.sample()
			np.append(features, sample, axis=0)
			s = self.__label(sample)
			print s
			print type(s)
			print s.shape
			raw_input('dataHandler.py line:46...')
			np.append(labels, self.__label(sample), axis=0)

		return (features, labels)

	def getInputSize(self):
		return len(self.features)

	def getOutputSize(self):
		return 1 #self.outSize[1]

'''
  Justin Chen
  9.29.16
  datahandler.py

  This module generates the data.

  Boston University Small Brain Project
  Policy Smoothing with Neural Networks
'''
import random
import numpy as np

class DataHandler:
    def __init__(self, feature_iters, output_size, label, batch_size=1):
        """
        Initializes data handler.

        Args:
            feature_iters: a sequence of iterators for each of the input
            features -- these iterators should either be infinite or effectively
            so, for repeated sample generation.

            output_size: number of expected outputs e.g. 1 for regression and >=
            for classification.

            label: a function that returns a label for a given input generated
            by the feature iterators.

            batch_size: size of which to generate batched samples of inputs.
        """
        self.feature_iters = feature_iters
        self.output_size   = output_size
        self.batch_size = batch_size
        self._label = label

    def sample(self):
        return np.array([next(feature_iter) for feature_iter in self.feature_iters])

    def get_batch(self):
        features = np.empty((self.batch_size, len(self.feature_iters)))
        labels = np.empty((self.batch_size, self.output_size))

        for b in range(self.batch_size):
            sample = self.sample()
            features[b] = sample
            labels[b] = self._label(*sample)

        return (features, labels)

    def get_input_size(self):
        return len(self.feature_iters)

    def get_output_size(self):
        return self.output_size

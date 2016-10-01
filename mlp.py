'''
  Justin Chen
  9.27.16
  mlp.py

  This class describes the neural network architecture

  Boston University Small Brain Project
  Policy Smoothing with Neural Networks
'''
import tensorflow as tf

class MultilayerPerceptron:

  def __init__(self, x, inputSize, outputSize):
      self.sample   = x
      self.n_input  = inputSize
      self.n_depth  = 2
      self.n_width  = 256
      self.n_output = outputSize
      self.weights  = {}
      self.biases   = {}

      # Generate layers
      # Assumes hidden layer sizes are all the same for now
      for h in range(0, self.n_depth):
        l_input =  self.n_input if h == 0 else self.n_width
        self.weights[str(h)] = tf.Variable(tf.random_normal([l_input, self.n_width]))
        self.biases[str(h)]  = tf.Variable(tf.random_normal([self.n_width]))

      self.weights['out'] = tf.Variable(tf.random_normal([self.n_width, self.n_output]))
      self.biases['out']  = tf.Variable(tf.random_normal([self.n_output]))

  def getNetwork(self):
    if self.sample != None and self.n_input > 0 and self.n_depth > 0 and self.n_width > 0 and self.n_output > 0:
      layer = None
      for h in range(0, self.n_depth):
        layer = self.sample if h == 0 else layer
        layer = tf.add(tf.matmul(layer, self.weights[str(h)]), self.biases[str(h)])
        layer = tf.nn.relu(layer)

      return tf.matmul(layer, self.weights['out']) + self.biases['out']
    return None
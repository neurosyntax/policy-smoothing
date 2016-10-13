'''
  Justin Chen
  9.27.16
  train.py

  Trains the neural network

  Boston University Small Brain Project
  Policy Smoothing with Neural Networks
'''

import tensorflow as tf
import datahandler as dh
import mlp

class Trainer:
  def __init__(self, feature_iters, output_size, label, data_size=10000,
               max_cost=-1, learning_rate=0.001, training_epochs=1000,
               batch_size=100, display_step=1):
    # training parameters
    self.data_size        = data_size
    self.max_cost        = max_cost
    self.learning_rate   = learning_rate
    self.training_epochs = training_epochs
    self.batch_size      = batch_size
    self.display_step    = display_step

    self.dataset = dh.DataHandler(
            feature_iters=feature_iters,
            output_size=output_size,
            label=label,
            batch_size=batch_size)

    # Network hyperparameters
    self.input_size  = self.dataset.get_input_size()
    self.output_size = self.dataset.get_output_size()

    # tf Graph input
    self.x = tf.placeholder("float", [None, self.input_size])
    self.y = tf.placeholder("float", [None, self.output_size])

    self.neural_net = mlp.MultilayerPerceptron(self.x, self.input_size, self.output_size).getNetwork()
    self.cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.neural_net, self.y))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
    self.init      = tf.initialize_all_variables()

  def train(self):
    with tf.Session() as sess:
        sess.run(self.init)

        # train
        for epoch in range(self.training_epochs):
            avg_cost = 0.0
            total_batch = int(self.data_size/self.batch_size)

            for i in range(total_batch):
              batch_x, batch_y = self.dataset.get_batch()
              # backprop
              _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
              avg_cost += c / total_batch

            if epoch % self.display_step == 0:
                print("epoch:", '%04d' % (epoch+1), "cost=",
                        "{:.9f}".format(avg_cost))
            if avg_cost < self.max_cost:
                break
        print("optimized...")

        # test model
        correct_prediction = tf.equal(tf.argmax(self.neural_net, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_x, test_y = self.dataset.get_batch()
        print(test_x)
        print(test_y)
        print("accuracy:", accuracy.eval({self.x: test_x, self.y: test_y}))

  def getNN(self):
    return self.neural_net

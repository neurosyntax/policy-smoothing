'''
  Justin Chen
  9.27.16
  train.py

  Trains the neural network

  Boston University Small Brain Project
  Policy Smoothing with Neural Networks
'''

import tensorflow as tf
from . import datahandler as dh
from . import mlp
import os
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell, seq2seq


class Trainer:
    def __init__(self, feature_iters, output_size, label, data_size=500000,
                 max_cost=-1, learning_rate=0.05, training_epochs=20, chkpt=None,
                 batch_size=100, memory_dim=100, display_step=1, delta=0.0035, save_interval=-1):
        # training parameters
        self.seq_length      = 10 # something
        self.vocab_size      = 10 # 0-9 
        self.depth           = 2
        self.lstm_size       = 256
        self.data_size       = data_size
        self.max_cost        = max_cost
        self.learning_rate   = learning_rate
        self.training_epochs = training_epochs
        self.batch_size      = batch_size
        self.memory_dim      = memory_dim
        self.display_step    = display_step
        self.delta           = delta
        self.save_interval   = save_interval
        self.chkpt           = chkpt

        self.dataset = dh.DataHandler(
                feature_iters=feature_iters,
                output_size=output_size,
                label=label,
                batch_size=batch_size)

        # Network
        self.input_size     = self.dataset.get_input_size()
        self.output_size    = self.dataset.get_output_size()
        self.x              = [tf.placeholder(tf.int32, shape=[None, self.lstm_size], name="input_%i"%t) for t in range(self.seq_length)]
        self.y              = [tf.placeholder(tf.int32, shape=[None, self.lstm_size], name="label_%i"%t) for t in range(self.seq_length)]
        self.weights        = [tf.ones_like(l, dtype=tf.float32) for l in self.y]
        self.decoder_input  = [tf.placeholder(tf.float32, shape=[None, self.lstm_size], name="decoder{0}".format(i)) for i in range(self.seq_length)]
        self.memory         = tf.zeros((self.batch_size, memory_dim))
        self.lstm_cell      = rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        self.multi_cell     = rnn_cell.MultiRNNCell([self.lstm_cell] * self.depth, state_is_tuple=True)
        (state, output)     = seq2seq.basic_rnn_seq2seq(self.x, self.decoder_input, self.multi_cell)
        self.decoder_output = state
        self.decoder_state  = output
        self.cost           = seq2seq.sequence_loss(self.decoder_output, self.y, self.weights, self.vocab_size)
        self.optimizer      = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        logdir = tempfile.mkdtemp()
        print(logdir)
        summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        
        self.init          = tf.initialize_all_variables()
        self.saver         = saver = tf.train.Saver()

    def train(self):
        with tf.Session() as sess:
            if self.chkpt and os.path.exists(self.chkpt):
                print("restoring model from checkpoint: {}".format(self.chkpt))
                self.saver.restore(sess, self.chkpt)
            else:
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
                     print("epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                 if avg_cost < self.max_cost:
                     break
                 if self.save_interval != -1 and  (epoch+1) % self.save_interval == 0:
                     save_path = self.saver.save(sess, self.chkpt)
                     print("network saved in %s" % save_path)
            print("optimized...")
            if self.chkpt:
                 save_path = self.saver.save(sess, self.chkpt)
                 print("network saved in %s" % save_path)


    def forward_pass(self, inputs):
        """ Takes a neural net, and returns the output of a single forward pass.

        Args:
            inputs: inputs to the neural net.
        """
        with tf.Session() as sess:
            if self.chkpt and os.path.exists(self.chkpt):
                print("Using model from checkpoint for forward pass: {}".format(self.chkpt))
                self.saver.restore(sess, self.chkpt)
            else:
                sess.run(self.init)
            feed_dict = {self.x: inputs}
            result = sess.run([self.neural_net], feed_dict=feed_dict)
        return result[0][0]


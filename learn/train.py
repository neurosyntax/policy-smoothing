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
from . import data_utils
from . import seq2seq_model
from . import data_utils
import os
import sys
import numpy as np


class Trainer:
    def __init__(self, feature_iters, output_size, label, data_size=500000, val_size=100000,
                 max_cost=-1, learning_rate=0.05, training_epochs=20, chkpt=None,
                 batch_size=64, memory_dim=100, display_step=1, delta=0.0035, save_interval=-1,
                 seq_length=3, enc_vocab_size=10, dec_vocab_size=10, depth=3, lstm_size=1024,
                 max_gradient=0.5, learning_decay=0.99, train_dir='/tmp', max_train_data_size=0,
                 use_fp16=False):
        # training parameters
        self.seq_length      = seq_length # something
        self.enc_vocab_size  = enc_vocab_size # 0-9 
        self.dec_vocab_size  = dec_vocab_size # 0-9
        self.depth           = depth
        self.lstm_size       = lstm_size
        self.max_gradient    = max_gradient # Clip gradients to this norm
        self.data_size       = data_size
        self.val_size        = val_size
        self.max_cost        = max_cost
        self.learning_rate   = learning_rate
        self.learning_decay  = learning_decay # Learning rate decays by this much
        self.training_epochs = training_epochs
        self.batch_size      = batch_size
        self.memory_dim      = memory_dim
        self.display_step    = display_step
        self.delta           = delta
        self.save_interval   = save_interval
        self.chkpt           = chkpt
        self.train_dir       = train_dir
        self.max_train_data_size = max_train_data_size # Limit on the size of training data (0: no limit)
        self.use_fp16        = use_fp16

        # set to 3 to 1 for now for input sequences of length 3 to a single value
        self.buckets = [(3, 1)] #[(5, 10), (10, 15), (20, 25), (40, 50)]


        self.dataset = dh.DataHandler(
                feature_iters=feature_iters,
                output_size=output_size,
                label=label,
                batch_size=batch_size)

        # Network
        '''
        self.input_size     = self.dataset.get_input_size()
        self.output_size    = self.dataset.get_output_size()
        self.x              = [tf.placeholder(tf.int32, shape=[None, self.seq_length], name="input_%i"%t) for t in range(self.enc_vocab_size)]
        self.y              = [tf.placeholder(tf.int32, shape=[None, self.lstm_size], name="label_%i"%t) for t in range(self.enc_vocab_size)]
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
        '''

        self.init          = tf.initialize_all_variables()
        #self.saver         = tf.train.Saver()

    def read_data(self, max_size):
      data_set = [[] for _ in self.buckets]
      source_ids = []
      target_ids = []
      #with tf.gfile.GFile(source_path, mode="r") as source_file:
      #  with tf.gfile.GFile(target_path, mode="r") as target_file:
      counter = 0
      for a in range(max_size):
        source, target = self.dataset.get_sample()
        while source.size > 0 and target.size > 0 and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [source] #[int(x) for x in source.split()]
          target_ids = [target] #[int(x) for x in target.split()]
          target_ids.append(data_utils.EOS_ID)

          # find appropriate sized bucket to put sequences into
          for bucket_id, (source_size, target_size) in enumerate(self.buckets):
            #print('source_ids:%d source_size:%d target_ids:%d target_size:%d',source_ids, source_size, target_ids, target_size)
            if len(source_ids) <= source_size and len(target_ids) <= target_size:
              print('Here')
              data_set[bucket_id].append([source_ids, target_ids])
              break
          source, target = self.dataset.get_sample() #source_file.readline(), target_file.readline()
      return data_set

    def create_model(self, session, forward_only):
        """Create translation model and initialize or load parameters in session."""
        dtype = tf.float16 if self.use_fp16 else tf.float32
        model = seq2seq_model.Seq2SeqModel(
            self.enc_vocab_size,
            self.dec_vocab_size,
            self.buckets,
            self.lstm_size,
            self.depth,
            self.max_gradient,
            self.batch_size,
            self.learning_rate,
            self.learning_decay,
            forward_only=forward_only,
            dtype=dtype)

        if self.chkpt and os.path.exists(self.chkpt):
          print("Using model from checkpoint for forward pass: {}".format(self.chkpt))
          self.saver.restore(sess, self.chkpt)
        else:
          print("Created model with fresh parameters.")
          session.run(tf.initialize_all_variables())
        return model

    def train(self):
        with tf.Session() as sess:
            model = self.create_model(sess, False)
            # Read data into buckets and compute their sizes.
            print ("Reading development and training data (limit: %d)."
                   % self.max_train_data_size)
            dev_set = self.read_data(self.val_size)
            train_set = self.read_data(self.data_size)
            print("train_set size: %d"%len(train_set))
            train_bucket_sizes = [len(train_set[b]) for b in range(len(self.buckets))]
            train_total_size = float(sum(train_bucket_sizes))

            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in range(len(train_bucket_sizes))]

            # This is the training loop.
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            while True:
              # Choose a bucket according to data distribution. We pick a random number
              # in [0, 1] and use the corresponding interval in train_buckets_scale.
              random_number_01 = np.random.random_sample()
              bucket_id = min([i for i in range(len(train_buckets_scale))
                               if train_buckets_scale[i] > random_number_01])

              # Get a batch and make a step.
              start_time = time.time()
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                  train_set, bucket_id)
              _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, False)
              step_time += (time.time() - start_time) / self.save_interval
              loss += step_loss / self.save_interval
              current_step += 1

              # Once in a while, we save checkpoint, print statistics, and run evals.
              if current_step % self.save_interval == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                  sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(self.train_dir, self.chkpt)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(self.buckets)):
                  if len(dev_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % (bucket_id))
                    continue
                  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                      dev_set, bucket_id)
                  _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                               target_weights, bucket_id, True)
                  eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                      "inf")
                  print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

            '''
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
            '''


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


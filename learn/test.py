import train

features =  [[(0,100000), (-4320,4320), (12340, 43210)], 1]
trainer  = train.Trainer(features=features, max_cost=100.00)
trainer.train()
nn = trainer.getNN()


#need to create a new session for each run to use the trained network
#prediction = sess.run(self.neuralNet, feed_dict={self.x: batch_x, self.y: batch_y})
import tensorflow as tf
import numpy as np
import random

class LinearRegression:

	def __init__(self):
		self.session = tf.Session()

	def train(self, X, y, learning_rate = 0.01, num_iters = 1000, batch_size=20, verbose=False):

		num_train, num_feature = X.shape

		self._construct_model(num_feature)
		loss = self._L2_loss(num_train)
		optimizer = self._SGD_optimizer(learning_rate, loss)

		saver = tf.train.Saver()
		init = tf.initialize_all_variables()

		# Launch the Graph
		with self.session as sess:
			sess.run(init)

			for epoch in range(1, num_iters+1):
				for i in range(batch_size):
					i = random.randrange(0, num_train)
					sess.run(optimizer, feed_dict={self.X: X[i].reshape(1, num_feature), self.Y: y[i]})

				if verbose and epoch % 50 == 0:
					print("Epoch: {:04d}  loss={:9f}".format(epoch, sess.run(loss, feed_dict={self.X: X, self.Y: y.reshape(num_train, 1)})) )

			print("Optimization Finished!")
			# Save the variables to disk.
			save_path = saver.save(sess, "./../tmp/model.ckpt")

	def _construct_model(self, num_feature):
		self.X = tf.placeholder("float", shape=(None, num_feature), name="X")
		self.Y = tf.placeholder("float", name="Y")

		self.W = tf.Variable(tf.random_normal([num_feature, 1], stddev=0.01, name="weight"))
		self.b = tf.Variable(tf.random_normal([1], name="bias"))

		self.pred_y = tf.add(tf.matmul(self.X, self.W), self.b, name="pred_y")

	def _L2_loss(self, num_train):
		return tf.reduce_sum(tf.square(self.Y - self.pred_y)) / (2*num_train)

	def _SGD_optimizer(self, learning_rate, loss):
		return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		
	def predict(self, X):
		saver = tf.train.Saver()
		num_feature = X.shape[0]
		
		with tf.Session() as sess:
			saver.restore(sess, "./../tmp/model.ckpt")
			return sess.run(self.pred_y, feed_dict={self.X: X.reshape(1, num_feature)})[0][0] # return flat value
	


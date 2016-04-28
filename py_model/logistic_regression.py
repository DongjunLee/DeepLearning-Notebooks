import numpy as np

class LogisticRegression(object):
	
	def __init__(self):
		self.W = None
		self.loss_history = []

	def train(self, X, y, learning_rate = 1e-3, regulizer=1e-5, num_iters=1000, batch_size=50, verbose=False):

		num_train, dim = X.shape
		num_classes = np.max(y) + 1

		# Lazy initialize W
		if self.W is None:
			self.W = 0.001 * np.random.randn(dim, num_classes)

		# SGD
		for epoch in range(num_iters):
			batch_xs = None
			batch_ys = None

			indices = np.random.choice(num_train, batch_size, replace=True)
			batch_xs = X[indices]
			batch_ys = y[indices]

			loss, dW = self.loss(batch_xs, batch_ys, regulizer)
			self.loss_history.append(loss)
			self.W += -learning_rate * dW

			if verbose and epoch % 50 == 0:
				print("Epoch : ", (epoch+1), " loss=", loss)

	def predict(self, X):
		scores = X.dot(self.W)
		return np.argmax(scores, axis=1)

	def loss(self, X, y, reg):
		
		# Initialize
		loss = 0.0
		dW = np.zeros_like(self.W)
		num_train = X.shape[0]

		scores = X.dot(self.W)
		scores -= np.max(scores, axis=1).reshape(num_train, 1)
		P = np.exp(scores)/np.reshape(np.sum(np.exp(scores), axis=1), (num_train, 1))
		loss = -np.sum(np.log(P[(range(num_train), y)]))
		
		loss /= num_train
		loss += 0.5 * reg * np.sum(self.W*self.W)

		P[(range(num_train), y)] = P[(range(num_train), y)] - 1
		dW = (1.0/num_train) * np.dot(X.T, P) + reg * self.W

		return loss, dW

	def get_loss_history(self):
		return self.loss_history

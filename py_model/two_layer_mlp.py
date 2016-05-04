import numpy as np

class NeuralNetwork(object):
	
	def __init__(self, l1_layer, l2_layer):
		self.l1 = {}
		self.l1['layer'] = l1_layer['layer']
		self.l1['activation'] = l1_layer['activation']
		self.__init_weight(self.l1)
		
		self.l2 = {}
		self.l2['layer'] = l2_layer['layer']
		self.__init_weight(self.l2)

		self.loss_history = []

	def __init_weight(self, layer):
		input_size, output_size = layer['layer']
		layer['W'] = 0.001 * np.random.randn(input_size, output_size)
		layer['b'] = 0.001 * np.random.randn(output_size)
		
	def loss(self, X, y, reg):
		loss = 0.0
		num_train = X.shape[0]

		_, output = self.forward(X)
		correct_scores = output[np.arange(num_train), y]
		loss = -np.sum(np.log(correct_scores))
		
		loss /= num_train
		loss += reg * (np.sum(self.l1['W']**2) + np.sum(self.l2['W']**2))

		return loss

	def forward(self, X):
		num_train = X.shape[0]

		l1_pre_act = X.dot(self.l1['W']) + self.l1['b']
		l1_act = self.l1['activation'](l1_pre_act)
		
		l2_pre_act = l1_act.dot(self.l2['W']) + self.l2['b']
		output = np.exp(l2_pre_act) / np.sum(np.exp(l2_pre_act), axis=1).reshape(num_train, 1)
		
		return l1_act, output

	def backward(self, X, y, learning_rate, reg):

		num_train = X.shape[0]
		grads = {}

		l1_act, output = self.forward(X)
		output[(np.arange(num_train), y)] = output[np.arange(num_train), y] - 1
		output /= num_train

		self.l2['W'] -= learning_rate * ( (l1_act.T).dot(output) + reg * (self.l2['W']) )
		self.l2['b'] -= learning_rate * np.sum(output, axis=0)
					
		dL1_act = output.dot(self.l2['W'].T)
			
		self.l1['W'] -= learning_rate * ( (X.T).dot(dL1_act) + reg * (self.l1['W']) ) 
		self.l1['b'] -= learning_rate * np.sum(dL1_act, axis=0)

	def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=1000, batch_size=50, verbose=False):

		num_train = X.shape[0]
		
		for epoch in range(num_iters):
			batch_xs = None
			batch_ys = None

			indices = np.random.choice(num_train, batch_size, replace=True)
			batch_xs = X[indices]
			batch_ys = y[indices]

			loss  = self.loss(batch_xs, batch_ys, reg)
			self.loss_history.append(loss)

			self.backward(batch_xs, batch_ys, learning_rate, reg)

			if verbose and epoch % 50 == 0:
				print("Epoch : ", (epoch+1), " loss=", loss)

	def predict(self, X):
		return np.argmax(self.forward(X)[1], axis=1)

import numpy as np

class N_LayerNetwork(object):
	
	def __init__(self, layers):
		self.layers = []
		for i in range(len(layers)):
			self.layers.append(layers[i])
			self.__init_weight(self.layers[i])
		self.layer_count = len(layers)
		self.loss_history = []

	def __init_weight(self, layer):
		input_size, output_size = layer['layer']
		layer['W'] = 0.001 * np.random.randn(input_size, output_size)
		layer['b'] = 0.001 * np.random.randn(output_size)
		
	def loss(self, X, y, reg):
		loss = 0.0
		num_train = X.shape[0]

		output = self.forward(X)[-1]
		correct_scores = output[np.arange(num_train), y]
		loss = -np.sum(np.log(correct_scores))
		
		loss /= num_train
		for i in range(self.layer_count):
			loss += reg * np.sum(self.layers[i]['W']**2)

		return loss

	def forward(self, X):
		num_train = X.shape[0]

		output = []
		for i in range(self.layer_count):	
			pre_act = self._pre_activation(X, i)
			if 'act_F' in self.layers[i]:	
				act = self.layers[i]['act_F'](pre_act)
				output.append(act)
				X = act
			else:
				output.append(pre_act)
				X = pre_act
		output[-1] = np.exp(output[-1]) / np.sum(np.exp(output[-1]), axis=1).reshape(num_train, 1)
		return output
	
	def _pre_activation(self, X, i):
		return np.dot(X, (self.layers[i]['W'])) + self.layers[i]['b']

	def backward(self, X, y, learning_rate, reg):

		num_train = X.shape[0]
		grads = {}

		output = self.forward(X)

		output[-1][(np.arange(num_train), y)] = output[-1][np.arange(num_train), y] - 1
		dL_act = output[-1]
		dL_act /= num_train

		for i in range(len(self.layers)-1, 0, -1):
			self.layers[i]['W'] -= learning_rate * ( (output[i-1].T).dot(dL_act) + reg * (self.layers[i]['W']) )
			self.layers[i]['b'] -= learning_rate * np.sum(dL_act, axis=0)
			dL_act = dL_act.dot(self.layers[i]['W'].T)
		self.layers[0]['W'] -= learning_rate * ( (X.T).dot(dL_act) + reg * (self.layers[0]['W']) )
		self.layers[0]['b'] -= learning_rate * np.sum(dL_act, axis=0)

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
		return np.argmax(self.forward(X)[-1], axis=1)

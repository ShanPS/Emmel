# Program to implement XOR gate using multi-layer Perceptron using numpy
import numpy as np

class XOR(object):
	# Initialize the model (weight matrices)
	def __init__(self, input_dim, hidden_dim, output_dim, lr = 0.25):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.lr = lr

		self.W1 = np.random.uniform(-0.5, 0.5, [input_dim, hidden_dim])
		self.b1 = np.zeros((hidden_dim,))
		self.W2 = np.random.uniform(-0.5, 0.5, [hidden_dim, output_dim])
		self.b2 = np.zeros((output_dim,))

		self.weights = [self.W1, self.b1, self.W2, self.b2]

	# Feedforward function: predicts the output using the model, given the input.
	def forward(self, x):
		self.z2 = np.dot(x, self.W1) + self.b1
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2) + self.b2
		self.a3 = self.sigmoid(self.z3)
		return self.a3

	# Activation function
	def sigmoid(self, x):
		return 1. / (1. + np.exp(-x))

	# Derivative of activation function
	def sigmoid_prime(self, x):
		return self.sigmoid(x) * (1 - self.sigmoid(x))

	# Loss function
	def mse(self, y_hat, y):
		return np.mean(0.5 * (y_hat - y)**2)

	# To get gradients
	def get_gradient(self, x, y):
		y_hat = self.forward(x)
		print("Loss : ", self.mse(y_hat, y))

		delta = (y_hat - y) * self.sigmoid_prime(self.z3)

		# Derivative of Loss functions with respect to weights
		dEdW2 = (np.dot(self.a2.reshape([-1,1]), delta)).reshape([-1,1])
		dEdb2 = delta
		dEdW1 = np.dot(x.reshape([-1,1]), (np.dot(delta, self.W2.T) * self.sigmoid_prime(self.z2)).reshape([1,-1]))
		dEdb1 = np.dot(delta, self.W2.T) * self.sigmoid_prime(self.z2)

		return [dEdW1, dEdb1, dEdW2, dEdb2]

	# Train the model
	def train(self, x, y):
		grads = self.get_gradient(x, y)

		for weight, grad in zip(self.weights, grads):
			weight -= grad * self.lr

np.random.seed(6699)
# Define the training data and outputs
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([[0.],[1.],[1.],[0.]])
# Set hyper-parameters
hidden_nodes = 4
epoch = 25000

nn = XOR(X.shape[1], hidden_nodes, y.shape[1])

for i in range(epoch):
	for j in range(4):
		nn.train(X[j],y[j])

print("\n\nPredicted output : ")
for i in X:
	print(nn.forward(i))

print("\nW1 :\n",nn.W1)
print("b1 : \n",nn.b1)
print("W2 : \n",nn.W2)
print("b2 : \n",nn.b2)
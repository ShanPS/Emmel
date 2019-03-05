# imports
import numpy as np
from keras.datasets import mnist


# declare the NN model class
class NeuralNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.025):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr

        # Hidden Layer Weights
        self.W1 = np.random.uniform(low=-1.0, high=1.0,
                                    size=(input_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,))

        # Output Layer Weights
        self.W2 = np.random.uniform(low=-1.0, high=1.0,
                                    size=(hidden_dim, output_dim))
        self.b2 = np.zeros((output_dim,))

        # List of Weights
        self.weights = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        # Hidden Layer preactivation
        self.z2 = np.dot(x, self.W1) + self.b1

        # Hidden Layer activation
        self.a2 = self.sigmoid(self.z2)

        # Output Layer preactivation
        self.z3 = np.dot(self.a2, self.W2) + self.b2

        # Output Layer activation = softmax (multiclass classification)
        self.a3 = self.softmax(self.z3)
        return self.a3

    def softmax(self, x):
        # for manual one by one output prediction done at the end
        if(x.ndim == 1):
            x = x - np.max(x).reshape((-1, 1))
            return np.exp(x)/np.sum(np.exp(x)).reshape((-1, 1))

        # Activation function for multiclass classification (Output layer)
        x = x - np.max(x, axis=1).reshape((-1, 1))
        return np.exp(x)/np.sum(np.exp(x), axis=1).reshape((-1, 1))

    def softmax_prime(self, x):
        return self.softmax(x) * (1 - self.softmax(x))

    def sigmoid(self, x):
        # Activation function for Hidden layers
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cross_entropy(self, y_hat, y):
        # Loss function
        y = y.argmax(axis=1)
        m = y.shape[0]
        p = y_hat
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def delta_cross_entropy(self, y, y_hat):
        y = y.argmax(axis=1)
        m = y.shape[0]
        grad = y_hat
        grad[range(m), y] -= 1
        grad = grad/m
        return grad

    def get_gradient(self, x, y):
        # Get the output of Neural Network
        y_hat = self.forward(x)

        # Pre compute common terms
        # delta = (y_hat - y) * self.softmax_prime(y_hat)
        delta = self.delta_cross_entropy(y, y_hat)
        a2_prime = self.sigmoid_prime(self.a2)

        # Compute W2 gradient
        dEdW2 = np.dot(self.a2.T, delta)
        # Compute b2 gradient
        dEb2 = np.sum(delta, axis=0)

        # Compute W1 gradient
        dEdW1 = np.dot(x.T, np.dot(delta, self.W2.T) * a2_prime)
        # Compute b1 gradient
        dEb1 = np.dot(delta, self.W2.T) * a2_prime
        dEb1 = np.sum(dEb1, axis=0)

        return [dEdW1, dEb1, dEdW2, dEb2]

    def train(self, x, y):
        # Compute gradients
        grads = self.get_gradient(x, y)

        # update weights using gradient descent
        for weight, grad in zip(self.weights, grads):
            weight -= grad * self.lr

    def accuracy(self, x, y):
        # Get the index of max_valued node for obtained output and real output
        x = np.argmax(x, axis=1)
        y = np.argmax(y, axis=1)

        # To keep track of correct outputs
        count = 0

        # Compare them
        for i in range(len(x)):
            if(x[i] == y[i]):
                count += 1

        # Return the percentage of correctness
        return (count / len(x)) * 100


# function for creating one hot vectors
def one_hot(y, nodes):
    # To change the shape of output array from (m,1) to (m,10)
    marker = y.ravel() * 10
    marker = marker.astype(int)
    result = np.zeros([y.shape[0], nodes])
    result[np.arange(y.shape[0]), marker] = 1
    return result

# Hyper-parameters
input_nodes = 784
hidden_nodes = 300
output_nodes = 10
epoch = 50
mini_batch = 10			# Adjust such that (mini_batch * steps = 60000)
steps = 6000

# Load data and pre-process the data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 784))/255
y_train = y_train.reshape(60000, 1)/10
y_train = one_hot(y_train, output_nodes)

x_test = x_test.reshape((10000, 784))/255
y_test = y_test.reshape(10000, 1)/10
y_test = one_hot(y_test, output_nodes)

# Create the model
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

# Train the model
for i in range(epoch):
    print("Iteration :", i+1)
    for j in range(steps):
        nn.train(x_train[mini_batch*j:mini_batch*j+mini_batch],
                 y_train[mini_batch*j:mini_batch*j+mini_batch])
    print('Loss:', nn.cross_entropy(nn.forward(x_train), y_train))
    # print(nn.forward(x_train))
    # print(y_train)

# np.set_printoptions(threshold=np.inf)
# print("\n\n\nW1")
# print(nn.W1)
# print("\n\n\nb1")
# print(nn.b1)
# print("\n\n\nW2")
# print(nn.W2)
# print("\n\n\nb2")
# print(nn.b2)
print("\n\n\nAccuracy =", nn.accuracy(nn.forward(x_test), y_test))


# For manual checking.!
while(True):
    print("press q to quit.! or Enter a number to check manually through test data (0-9999)")
    x = input()
    if(x == 'q'):
        break
    print(nn.forward(x_test[int(x)]))
    print(y_test[int(x)])

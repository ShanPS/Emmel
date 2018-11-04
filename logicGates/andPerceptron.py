# Program to implement 2 input AND gate using single layer Perceptron using numpy
# Just for understanding weight updation and feedforwarding. (No specific structure used)
import numpy as np

np.random.seed(1)
# Initialize weights with random vales
weight = np.random.random([2,1])
# declare data and output
data = np.array([(0,0),(0,1),(1,0),(1,1)])
output = np.array([(0),(0),(0),(1)])		# output = np.array([(0),(1),(1),(1)]) ---- for OR gate
# declare hyper-parameters
alpha = 0.15
thresh = 0.1

# Function to return result for an input using the model
def perceptron(x):
	result = x.dot(weight)
	if(sigmoid(result) < thresh):
		obtainedOut = 0
	else:
		obtainedOut = 1
	return obtainedOut

# Function to train the model
def train():
	global thresh
	errorNum = 0
	for i in range(25):
		errorNum = 0
		for j in range(4):
			obtainedOut = perceptron(data[j])
			# Update the weights
			weight[0] += alpha*(output[j]-obtainedOut)*data[j][0]
			weight[1] += alpha*(output[j]-obtainedOut)*data[j][1]
			thresh +=  alpha*(obtainedOut-output[j])
			# To get the error rate
			errorNum += abs(output[j]-obtainedOut)
		print("\nIteration:{}  Loss = {}".format(i+1,errorNum/4))
		print("weights =" , weight)
		print("threshold =" , thresh)

# Activation function
def sigmoid(n):
	return 1/(1+np.exp(-n))



train()

print("\n\nRESULTS:")
print("0 and 0 : {}".format(perceptron(data[0])))
print("0 and 1 : {}".format(perceptron(data[1])))
print("1 and 0 : {}".format(perceptron(data[2])))
print("1 and 1 : {}".format(perceptron(data[3])))

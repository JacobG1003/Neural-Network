import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from keras.datasets import mnist
warnings.filterwarnings('ignore')

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_prime(x): return x * (1 - x)
def error(output_activations, output_true): return 0.5 * np.square(output_activations-output_true)
def error_prime(output_activations, output_true): return (output_activations-output_true)
def tanh(x): return np.tanh(x)
def tanh_prime(x): return 1 - x ** 2
def relu(x): return np.max(0, x)
def relu_prime(x): return 1 if x > 0 else 0

def activation(x): return sigmoid(x)
def activation_prime(x): return sigmoid_prime(x)

class Network():
    def __init__(self, layer_sizes):
        self.error = []
        self.layers = []
        for i in range(1,len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i]))
        
    def forwardpropagate(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_step(inputs)
        return inputs

    def backpropagate(self, inputs, target, learning_rate):
        output = self.forwardpropagate(inputs)
        self.layers[-1].deltas = (self.layers[-1].activations - target) * sigmoid_prime(self.layers[-1].activations)
        for i in range(len(self.layers)-2, -1,-1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer.deltas = np.dot(next_layer.deltas, next_layer.weights.T) * sigmoid_prime(layer.activations)
        for i in range(len(self.layers)-1, 0,-1):
            self.layers[i].apply_gradient(self.layers[i - 1].activations, learning_rate)
        self.layers[0].apply_gradient(inputs, learning_rate) 
    
        return output


    def train(self, inputs, target_outputs, iterations, learning_rate):
        start_time = time.time()
        for i in range(iterations):
            iteration_error = 0
            for j in range(len(inputs)):
                outputs = self.backpropagate(inputs[j], target_outputs[j], learning_rate)
                iteration_error += np.sum(error(target_outputs[j],outputs))
            iteration_error /= len(inputs)
            self.error.append(iteration_error)
            if i != iterations - 1:
                print(f"Progress: [{i} / {iterations}] error: {iteration_error:.5e} ", end="\r", flush=True)
            else:
                print(f"Progress: [{iterations} / {iterations}] error: {iteration_error:.5e}", end="\n", flush=True)
        end_time = time.time()
        total_time = end_time - start_time 
        print(f"Elapsed time: {total_time:.2f} seconds")


class Layer():
    def __init__(self, num_prev_neurons, num_neurons):
        self.weights = np.random.randn(num_prev_neurons, num_neurons)
        self.biases = np.random.randn(num_neurons)
        self.deltas = np.zeros((num_neurons)) 
        self.activations = np.zeros((num_neurons)) 
        
    def forward_step(self, inputs):
        self.activations = sigmoid(np.dot(inputs, self.weights) + self.biases)
        return self.activations
    
    def apply_gradient(self, inputs, learning_rate):
        self.biases -= learning_rate * self.deltas
        self.weights -= learning_rate * self.deltas * inputs[:, None]

    

def main():
    #Initialize Paramaters and Create network
    networkShape = [784,75,10]
    iterations = 10
    training_rate = 0.75
    net = Network(networkShape)
    print(networkShape)
    #Import the Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #Format Training Data
    numTrainPoints = 60000
    x_train = x_train
    x_train = x_train[:numTrainPoints].reshape(x_train.shape[0], -1)
    x_train = x_train / 255.0
    y_train = np.eye(10)[y_train[:numTrainPoints]]
    #Train network
    net.train(x_train, y_train, iterations, training_rate)
    #Format Test Data
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = x_test.astype('float32') / 255
    y_test = np.eye(10)[y_test]
    inputs_test = x_test[:10000]
    outputs_test = y_test[:10000]
    #Test the network
    correct = 0
    for i in range(len(inputs_test)):
        correct += np.argmax(net.forwardpropagate(inputs_test[i])) == np.argmax(outputs_test[i])
    print(f"Accuracy: {correct/len(inputs_test)*100:.2f}%")
    # plot the error over time
    plt.plot(net.error)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()

if __name__ == "__main__":
    main()
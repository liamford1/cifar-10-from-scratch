import numpy as np
from layers import *

class CNN:
    def __init__(self):
        self.conv1_kernels = np.random.randn(32, 3, 3, 3) * 0.1
        self.conv2_kernels = np.random.randn(64, 3, 3, 32) * 0.1

        self.weights1 = np.random.randn(4096, 256) * 0.1
        self.weights2 = np.random.randn(256, 10) * 0.1

        self.biases1 = np.random.randn(256) * 0.1
        self.biases2 = np.random.randn(10) * 0.1

    def forward(self, x):
        self.x_input = x

        #First Convolutional Layer
        self.padded_input = padding(x)
        self.conv1 = conv(self.padded_input, self.conv1_kernels)
        self.relu1 = relu(self.conv1)
        self.pool1 = max_pool(self.relu1)

        #Second Convolutional Layer
        self.second_padding = padding(self.pool1)
        self.conv2 = conv(self.second_padding, self.conv2_kernels)
        self.relu2 = relu(self.conv2)
        self.pool2 = max_pool(self.relu2)

        #Flatten
        self.flattened = flatten(self.pool2)

        #First Dense Layer
        self.dense1 = dense(self.flattened, self.weights1, self.biases1)
        self.dense_relu = relu(self.dense1)

        #Second (Output) Dense Layer
        self.dense2 = dense(self.dense_relu, self.weights2, self.biases2)

        self.probs = softmax(self.dense2)

        return self.probs

    def backprop(self, true_labels):

        #Second Dense Layer Gradients
        self.output_gradients = softmax_cross_entropy_gradient(self.probs, true_labels)
        self.d_weights2 = np.outer(self.dense_relu, self.output_gradients)
        self.d_biases2 = self.output_gradients
        self.d_dense_relu = self.weights2 @ self.output_gradients
        self.d_dense1 = self.d_dense_relu * relu_deriv(self.dense1)

        #First Dense Layer Gradients
        self.d_weights1 = np.outer(self.flattened, self.d_dense1)
        self.d_biases1 = self.d_dense1
        self.d_flattened = self.weights1 @ self.d_dense1
        self.d_pool2 = np.reshape(self.d_flattened, self.pool2.shape)

        #Second Max Pooling Gradients
        self.d_relu2 = max_pool_backward(self.relu2, self.d_pool2)
        self.d_conv2 = self.d_relu2 * relu_deriv(self.conv2)

        #Second Convolutional Layer Gradients
        self.d_conv2_kernels = conv_kernels_backward(self.second_padding, self.d_conv2, self.conv2_kernels.shape)
        self.d_second_padding = conv_input_backward(self.d_conv2, self.conv2_kernels, self.second_padding.shape)

        #Remove Second Padding
        self.d_pool1 = self.d_second_padding[1:-1, 1:-1, :]
        
        #First Max Pooling Gradients
        self.d_relu1 = max_pool_backward(self.relu1, self.d_pool1)
        self.d_conv1 = self.d_relu1 * relu_deriv(self.conv1)
        
        #First Convolutional Layer Gradients
        self.d_conv1_kernels = conv_kernels_backward(self.padded_input, self.d_conv1, self.conv1_kernels.shape)
        self.d_padded_input = conv_input_backward(self.d_conv1, self.conv1_kernels, self.padded_input.shape)

        #Remove First Padding
        self.d_input = self.d_padded_input[1:-1, 1:-1, :]
    
    def update_weights(self, lr):
        self.conv1_kernels -= lr * self.d_conv1_kernels
        self.conv2_kernels -= lr * self.d_conv2_kernels

        self.weights1 -= lr * self.d_weights1
        self.weights2 -= lr * self.d_weights2

        self.biases1 -= lr * self.d_biases1
        self.biases2 -= lr * self.d_biases2
    
    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)

    def compute_loss(self, predictions, true_labels):
        return cross_entropy_loss(predictions, true_labels)
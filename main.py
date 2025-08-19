import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from data_loader import load_cifar10_data

x_train, y_train, x_test, y_test = load_cifar10_data()
print(f"Training data shape: {x_train.shape}")

def padding(input):
    return np.pad(input, pad_width=((1,1), (1,1), (0,0)), mode='constant', constant_values=0)

def conv(input, kernels):
    num_kernels = kernels.shape[0]
    h, w = input.shape[0] - 2, input.shape[1] - 2
    output = np.zeros((h, w, num_kernels))

    for i in range(h):
        for j in range(w):
            for k in range(num_kernels):
                patch = input[i:i+3, j:j+3, :]
                output[i, j, k] = np.sum(patch * kernels[k])
    return output

def max_pool(input):
    h, w, c = input.shape
    output = np.zeros((h//2, w//2, c))

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            for k in range(c):
                patch = input[i:i+2, j:j+2, k]
                output[i//2, j//2, k] = np.max(patch)
    return output

def flatten(input):
    return input.flatten()

def dense(input, w, b):
    return input @ w + b

def relu(input):
    return np.maximum(0, input)

def softmax(input):
    exp_x = np.exp(input - np.max(input))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(preds, true):
    return -np.sum(true * np.log(preds + 1e-15))

def softmax_cross_entropy_gradient(preds, true):
    return preds - true

def relu_deriv(x):
    return (x > 0).astype(int)

def max_pool_backward(input, grad_output):

        h, w, c = input.shape
        grad_input = np.zeros(input.shape)

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                for k in range(c):
                    patch = input[i:i+2, j:j+2, k]
                    max_num = np.argmax(patch)
                    row = max_num // 2
                    col = max_num % 2
                    grad_input[i + row, j + col, k] = grad_output[i//2, j//2, k]

        return grad_input

def conv_kernels_backward(input, grad_output, kernel_shape):
    grads = np.zeros(kernel_shape)
    h, w = input.shape[0] - 2, input.shape[1] - 2
    
    for i in range(h):
        for j in range(w):
            for k in range(kernel_shape[0]):
                patch = input[i:i+3, j:j+3, :]
                grads[k] += patch * grad_output[i, j, k]

    return grads

def conv_input_backward(grad_output, kernels, input_shape):
    grad_input = np.zeros(input_shape)
    h, w = grad_output.shape[0], grad_output.shape[1]

    for i in range(h):
        for j in range(w):
            for k in range(kernels.shape[0]):
                kernel = kernels[k]
                grad_input[i:i+3, j:j+3, :] += kernel * grad_output[i, j, k]

    return grad_input

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

    def compute_loss(self, preds, true_labels):
        return cross_entropy_loss(preds, true_labels)

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

def test_cnn():
    print("TESTING")

    cnn = CNN()

    sample_x = x_train[0]
    sample_y = y_train[0]


    print(f"Input image shape: {sample_x.shape}")
    print(f"True label (one-hot): {sample_y}")
    print(f"True class: {np.argmax(sample_y)}")


    print("Before training:")
    predictions = cnn.forward(sample_x)
    loss = cnn.compute_loss(predictions, sample_y)
    print(f"Loss: {loss:.4f}")
    print(f"Confidence in true class: {predictions[np.argmax(sample_y)]:.4f}")

    learning_rate = 0.01

    for i in range(10):
        predictions = cnn.forward(sample_x)
        cnn.backprop(sample_y)
        cnn.update_weights(learning_rate)
        
        if i % 2 == 0:
            loss = cnn.compute_loss(predictions, sample_y)
            print(f"Step {i}: Loss = {loss:.4f}")
    
    print("\nAfter training:")
    predictions = cnn.forward(sample_x)
    loss = cnn.compute_loss(predictions, sample_y)
    print(f"Final loss: {loss:.4f}")
    print(f"Final confidence: {predictions[np.argmax(sample_y)]:.4f}")


if __name__ == "__main__":
    cnn = test_cnn()


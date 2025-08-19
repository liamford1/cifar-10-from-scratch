import numpy as np
from layers import *

class CNN:
    def __init__(self):
        self.conv1_kernels = np.random.randn(32, 3, 3, 3) * np.sqrt(2.0 / (3 * 3 * 3))
        self.conv2_kernels = np.random.randn(64, 3, 3, 32) * np.sqrt(2.0 / (3 * 3 * 32))

        self.weights1 = np.random.randn(4096, 256) * np.sqrt(2.0 / 4096)
        self.weights2 = np.random.randn(256, 10) * np.sqrt(2.0 / 256)

        self.biases1 = np.zeros(256)
        self.biases2 = np.zeros(10)

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
    
class BatchCNN:
    def __init__(self):
        # Xavier/Glorot initialization for better training
        self.conv1_kernels = np.random.randn(32, 3, 3, 3) * np.sqrt(2.0 / (3 * 3 * 3))
        self.conv2_kernels = np.random.randn(64, 3, 3, 32) * np.sqrt(2.0 / (3 * 3 * 32))

        self.weights1 = np.random.randn(4096, 256) * np.sqrt(2.0 / 4096)
        self.weights2 = np.random.randn(256, 10) * np.sqrt(2.0 / 256)

        self.biases1 = np.zeros(256)  # Initialize biases to zero
        self.biases2 = np.zeros(10)   # Initialize biases to zero

    def forward(self, x):
        self.x_input = x

        #First Convolutional Layer
        padded = batch_padding(x)
        self.conv1 = batch_conv(padded, self.conv1_kernels)
        np.maximum(self.conv1, 0, out=self.conv1)
        self.pool1 = batch_max_pool(self.conv1)
        
        #Second Convolutional Layer
        self.second_padding = batch_padding(self.pool1)
        self.conv2 = batch_conv(self.second_padding, self.conv2_kernels)
        self.relu2 = batch_relu(self.conv2)
        self.pool2 = batch_max_pool(self.relu2)

        #Flatten
        self.flattened = batch_flatten(self.pool2)
        
        #First Dense Layer
        self.dense1 = batch_dense(self.flattened, self.weights1, self.biases1)
        self.dense_relu = batch_relu(self.dense1)

        #Second (Output) Dense Layer
        self.dense2 = batch_dense(self.dense_relu, self.weights2, self.biases2)

        self.probs = batch_softmax(self.dense2)

        return self.probs
    
    def backprop(self, true_labels):
        batch_size = self.probs.shape[0]
        
        # Second Dense Layer Gradients
        self.output_gradients = self.probs - true_labels  # Shape: (batch_size, 10)
        
        # For weights2: average of outer products across batch
        self.d_weights2 = np.zeros_like(self.weights2)
        for i in range(batch_size):
            self.d_weights2 += np.outer(self.dense_relu[i], self.output_gradients[i])
        self.d_weights2 /= batch_size
        
        # For biases2: average across batch
        self.d_biases2 = np.mean(self.output_gradients, axis=0)
        
        # For dense_relu gradients: batch matrix multiplication
        self.d_dense_relu = self.output_gradients @ self.weights2.T  # Shape: (batch_size, 256)
        self.d_dense1 = self.d_dense_relu * relu_deriv(self.dense1)
        
        # First Dense Layer Gradients
        self.d_weights1 = np.zeros_like(self.weights1)
        for i in range(batch_size):
            self.d_weights1 += np.outer(self.flattened[i], self.d_dense1[i])
        self.d_weights1 /= batch_size
        
        self.d_biases1 = np.mean(self.d_dense1, axis=0)
        self.d_flattened = self.d_dense1 @ self.weights1.T  # Shape: (batch_size, 4096)
        
        # Reshape back to conv shape
        self.d_pool2 = self.d_flattened.reshape(self.pool2.shape)
        
        # For conv layers, we'll process each sample in the batch individually
        # (This is simpler than creating full batch backward functions)
        self.d_relu2 = np.zeros_like(self.relu2)
        self.d_conv2 = np.zeros_like(self.conv2)
        self.d_conv2_kernels = np.zeros_like(self.conv2_kernels)
        self.d_second_padding = np.zeros_like(self.second_padding)
        
        for i in range(batch_size):
            # Second max pool backward
            d_relu2_i = max_pool_backward(self.relu2[i], self.d_pool2[i])
            self.d_relu2[i] = d_relu2_i
            
            # Second conv backward
            d_conv2_i = d_relu2_i * relu_deriv(self.conv2[i])
            self.d_conv2[i] = d_conv2_i
            
            # Conv2 kernel gradients (accumulate across batch)
            self.d_conv2_kernels += conv_kernels_backward(self.second_padding[i], d_conv2_i, self.conv2_kernels.shape)
            
            # Conv2 input gradients
            d_second_padding_i = conv_input_backward(d_conv2_i, self.conv2_kernels, self.second_padding[i].shape)
            self.d_second_padding[i] = d_second_padding_i
        
        # Remove second padding (for each sample in batch)
        self.d_pool1 = self.d_second_padding[:, 1:-1, 1:-1, :]
        
        # First conv layer - same pattern
        self.d_relu1 = np.zeros_like(self.relu1)
        self.d_conv1 = np.zeros_like(self.conv1)
        self.d_conv1_kernels = np.zeros_like(self.conv1_kernels)
        self.d_padded_input = np.zeros_like(self.padded_input)
        
        for i in range(batch_size):
            # First max pool backward
            d_relu1_i = max_pool_backward(self.relu1[i], self.d_pool1[i])
            self.d_relu1[i] = d_relu1_i
            
            # First conv backward
            d_conv1_i = d_relu1_i * relu_deriv(self.conv1[i])
            self.d_conv1[i] = d_conv1_i
            
            # Conv1 kernel gradients (accumulate across batch)
            self.d_conv1_kernels += conv_kernels_backward(self.padded_input[i], d_conv1_i, self.conv1_kernels.shape)
            
            # Conv1 input gradients
            d_padded_input_i = conv_input_backward(d_conv1_i, self.conv1_kernels, self.padded_input[i].shape)
            self.d_padded_input[i] = d_padded_input_i
        
        # Average conv1 kernel gradients across batch
        self.d_conv1_kernels /= batch_size
        
        # Remove first padding
        self.d_input = self.d_padded_input[:, 1:-1, 1:-1, :]
    
    def update_weights(self, lr):
        clip_norm = 1.0
        for grad in [self.d_conv1_kernels, self.d_conv2_kernels, self.d_weights1, self.d_weights2, self.d_biases1, self.d_biases2]:
            norm = np.linalg.norm(grad)
            if norm > clip_norm:
                grad *= clip_norm / norm

        self.conv1_kernels -= lr * self.d_conv1_kernels
        self.conv2_kernels -= lr * self.d_conv2_kernels
        self.weights1 -= lr * self.d_weights1
        self.weights2 -= lr * self.d_weights2
        self.biases1 -= lr * self.d_biases1
        self.biases2 -= lr * self.d_biases2

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)  # Return predictions for each sample in batch

    def compute_loss(self, predictions, true_labels):
        return -np.mean(np.sum(true_labels * np.log(predictions + 1e-15), axis=1))
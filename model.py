import numpy as np
from layers import *
from config import ModelConfig

class BaseCNN:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialize_weights()
    
    def _initialize_weights(self):
        conv1_input_channels = self.config.input_shape[2]

        self.conv1_kernels = np.random.randn(
            self.config.conv1_filters, 
            self.config.kernel_size, 
            self.config.kernel_size, 
            conv1_input_channels
        ) * np.sqrt(2.0 / (self.config.kernel_size * self.config.kernel_size * conv1_input_channels))

        self.conv2_kernels = np.random.randn(
            self.config.conv2_filters, 
            self.config.kernel_size, 
            self.config.kernel_size, 
            self.config.conv1_filters
        ) * np.sqrt(2.0 / (self.config.kernel_size * self.config.kernel_size * self.config.conv1_filters))

        flattened_size = 4096

        self.weights1 = np.random.randn(flattened_size, self.config.dense1_units) * np.sqrt(2.0 / flattened_size)
        self.weights2 = np.random.randn(self.config.dense1_units, self.config.dense2_units) * np.sqrt(2.0 / self.config.dense1_units)

        self.biases1 = np.zeros(self.config.dense1_units)
        self.biases2 = np.zeros(self.config.dense2_units)

    def compute_loss(self, predictions, true_labels):
        if predictions.ndim == 1:
            return -np.sum(true_labels * np.log(predictions + 1e-15))
        else:
            return -np.mean(np.sum(true_labels * np.log(predictions + 1e-15), axis=1))

    def update_weights(self, lr):
        self.conv1_kernels -= lr * self.d_conv1_kernels
        self.conv2_kernels -= lr * self.d_conv2_kernels

        self.weights1 -= lr * self.d_weights1
        self.weights2 -= lr * self.d_weights2

        self.biases1 -= lr * self.d_biases1
        self.biases2 -= lr * self.d_biases2

class CNN(BaseCNN):
    def forward(self, x):
        self.x_input = x

        self.padded_input = padding(x)
        self.conv1 = conv(self.padded_input, self.conv1_kernels)
        self.relu1 = relu(self.conv1)
        self.pool1 = max_pool(self.relu1)

        self.second_padding = padding(self.pool1)
        self.conv2 = conv(self.second_padding, self.conv2_kernels)
        self.relu2 = relu(self.conv2)
        self.pool2 = max_pool(self.relu2)

        self.flattened = flatten(self.pool2)

        self.dense1 = dense(self.flattened, self.weights1, self.biases1)
        self.dense_relu = relu(self.dense1)

        self.dense2 = dense(self.dense_relu, self.weights2, self.biases2)

        self.probs = softmax(self.dense2)

        return self.probs

    def backprop(self, true_labels):
        self.output_gradients = self.probs - true_labels
        self.d_weights2 = np.outer(self.dense_relu, self.output_gradients)
        self.d_biases2 = self.output_gradients
        self.d_dense_relu = self.weights2 @ self.output_gradients
        self.d_dense1 = self.d_dense_relu * relu_deriv(self.dense1)

        self.d_weights1 = np.outer(self.flattened, self.d_dense1)
        self.d_biases1 = self.d_dense1
        self.d_flattened = self.weights1 @ self.d_dense1
        self.d_pool2 = np.reshape(self.d_flattened, self.pool2.shape)

        self.d_relu2 = max_pool_backward(self.relu2, self.d_pool2)
        self.d_conv2 = self.d_relu2 * relu_deriv(self.conv2)

        self.d_conv2_kernels = conv_kernels_backward(self.second_padding, self.d_conv2, self.conv2_kernels.shape)
        self.d_second_padding = conv_input_backward(self.d_conv2, self.conv2_kernels, self.second_padding.shape)

        self.d_pool1 = self.d_second_padding[1:-1, 1:-1, :]
        
        self.d_relu1 = max_pool_backward(self.relu1, self.d_pool1)
        self.d_conv1 = self.d_relu1 * relu_deriv(self.conv1)
        
        self.d_conv1_kernels = conv_kernels_backward(self.padded_input, self.d_conv1, self.conv1_kernels.shape)
        self.d_padded_input = conv_input_backward(self.d_conv1, self.conv1_kernels, self.padded_input.shape)

        self.d_input = self.d_padded_input[1:-1, 1:-1, :]
    
    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)
    
class BatchCNN(BaseCNN):
    def forward(self, x):
        self.x_input = x

        self.padded_input = batch_padding(x)
        self.conv1 = batch_conv(self.padded_input, self.conv1_kernels)
        self.relu1 = batch_relu(self.conv1)
        self.pool1 = batch_max_pool(self.relu1)
        
        self.second_padding = batch_padding(self.pool1)
        self.conv2 = batch_conv(self.second_padding, self.conv2_kernels)
        self.relu2 = batch_relu(self.conv2)
        self.pool2 = batch_max_pool(self.relu2)

        self.flattened = batch_flatten(self.pool2)
        
        self.dense1 = batch_dense(self.flattened, self.weights1, self.biases1)
        self.dense_relu = batch_relu(self.dense1)

        self.dense2 = batch_dense(self.dense_relu, self.weights2, self.biases2)

        self.probs = batch_softmax(self.dense2)

        return self.probs
    
    def backprop(self, true_labels):
        batch_size = self.probs.shape[0]
        
        self.output_gradients = self.probs - true_labels
        
        self.d_weights2 = (self.dense_relu.T @ self.output_gradients) / batch_size
        
        self.d_biases2 = np.mean(self.output_gradients, axis=0)
        
        self.d_dense_relu = self.output_gradients @ self.weights2.T
        self.d_dense1 = self.d_dense_relu * relu_deriv(self.dense1)
        
        self.d_weights1 = (self.flattened.T @ self.d_dense1) / batch_size
        
        self.d_biases1 = np.mean(self.d_dense1, axis=0)
        self.d_flattened = self.d_dense1 @ self.weights1.T
        
        self.d_pool2 = self.d_flattened.reshape(self.pool2.shape)
        
        self.d_relu2 = np.zeros_like(self.relu2)
        self.d_conv2 = np.zeros_like(self.conv2)
        self.d_conv2_kernels = np.zeros_like(self.conv2_kernels)
        self.d_second_padding = np.zeros_like(self.second_padding)
        
        # Use vectorized batch functions
        self.d_relu2 = batch_max_pool_backward(self.relu2, self.d_pool2)
        self.d_conv2 = self.d_relu2 * relu_deriv(self.conv2)
        self.d_second_padding = batch_conv_input_backward(self.d_conv2, self.conv2_kernels, self.second_padding.shape)
        
        self.d_conv2_kernels = batch_conv_kernels_backward(self.second_padding, self.d_conv2, self.conv2_kernels.shape)
        
        self.d_pool1 = self.d_second_padding[:, 1:-1, 1:-1, :]
        
        self.d_relu1 = np.zeros_like(self.relu1)
        self.d_conv1 = np.zeros_like(self.conv1)
        self.d_conv1_kernels = np.zeros_like(self.conv1_kernels)
        self.d_padded_input = np.zeros_like(self.padded_input)
        
        # Use vectorized batch functions
        self.d_relu1 = batch_max_pool_backward(self.relu1, self.d_pool1)
        self.d_conv1 = self.d_relu1 * relu_deriv(self.conv1)
        self.d_padded_input = batch_conv_input_backward(self.d_conv1, self.conv1_kernels, self.padded_input.shape)
        
        self.d_conv1_kernels = batch_conv_kernels_backward(self.padded_input, self.d_conv1, self.conv1_kernels.shape)
        
        self.d_input = self.padded_input[:, 1:-1, 1:-1, :]
    
    def update_weights(self, lr):
        clip_norm = 1.0
        for grad in [self.d_conv1_kernels, self.d_conv2_kernels, self.d_weights1, self.d_weights2, self.d_biases1, self.d_biases2]:
            norm = np.linalg.norm(grad)
            if norm > clip_norm:
                grad *= clip_norm / norm

        super().update_weights(lr)

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
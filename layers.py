import numpy as np

#Forward Pass Functions
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

def dense(input, w, b):
    return input @ w + b

def relu(input):
    return np.maximum(0, input)

def softmax(input):
    exp_x = np.exp(input - np.max(input))
    return exp_x / np.sum(exp_x)

def padding(input):
    return np.pad(input, pad_width=((1,1), (1,1), (0,0)), mode='constant', constant_values=0)

def flatten(input):
    return input.flatten()

#BackProp Functions
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

def relu_deriv(x):
    return (x > 0).astype(int)

#Loss Functions
def cross_entropy_loss(preds, true):
    return -np.sum(true * np.log(preds + 1e-15))

def softmax_cross_entropy_gradient(preds, true):
    return preds - true
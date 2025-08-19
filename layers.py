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
    h, w = input.shape[0] - 2, input.shape[1] - 2
    patches = np.lib.stride_tricks.sliding_window_view(input, (3, 3, input.shape[2]), axis=(0, 1, 2)).reshape(h, w, 3*3*input.shape[2])
    grad_flat = grad_output.reshape(h*w, grad_output.shape[2])
    grads = (patches.reshape(h*w, -1).T @ grad_flat).T

    return grads.reshape(kernel_shape)

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

#Batch Functions
def batch_conv(input_batch, kernels):
    batch_size, h, w, c = input_batch.shape
    num_kernels = kernels.shape[0]
    output = np.zeros((batch_size, h, w, num_kernels))

    patches = np.lib.stride_tricks.sliding_window_view(input_batch, (3, 3, c), axis=(1, 2, 3)).reshape(batch_size, h-2, w-2, 3*3*c)
    kernels_flat = kernels.reshape(num_kernels, 3*3*c)
    output = patches @ kernels_flat.T

    return output.reshape(batch_size, h-2, w-2, num_kernels)

def batch_conv_kernels_backward(input_batch, grad_output_batch, kernel_shape):
    batch_size, h, w, c = input_batch.shape
    num_kernels = grad_output_batch.shape[3]

    patches = np.lib.stride_tricks.sliding_window_view(input_batch, (3, 3, c), axis=(1, 2, 3)).reshape(batch_size, h-2, w-2, 3*3*c)
    grad_flat = grad_output_batch.reshape(batch_size, (h-2)*(w-2), num_kernels)

    grads = np.zeros((batch_size, 3*3*c, num_kernels))
    for b in range(batch_size):
        grads[b] = patches[b].reshape((h-2)*(w-2), 3*3*c).T @ grad_flat[b]
    
    return np.mean(grads, axis=0).reshape(kernel_shape)


def batch_max_pool(input_batch):
    batch_size, h, w, c = input_batch.shape

    reshaped = input_batch.reshape(batch_size, h//2, 2, w//2, 2, c)

    return np.max(reshaped, axis=(2, 4))

def batch_dense(input_batch, w, b):
    if not input_batch.flags['C_CONTIGUOUS']:
        input_batch = np.ascontiguousarray(input_batch)
    return input_batch @ w + b

def batch_relu(input_batch):
    return np.maximum(0, input_batch)

def batch_softmax(input_batch):
    exp_x = np.exp(input_batch - np.max(input_batch, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def batch_padding(input_batch):
    return np.pad(input_batch, pad_width=((0,0), (1,1), (1,1), (0,0)), mode='constant', constant_values=0)

def batch_flatten(input_batch):
    return input_batch.reshape(input_batch.shape[0], -1)
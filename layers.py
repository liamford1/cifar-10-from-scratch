import numpy as np
from scipy import signal

#Forward Pass Functions
def conv(input, kernels):
    num_kernels = kernels.shape[0]
    h, w = input.shape[0] - 2, input.shape[1] - 2
    
    patches = np.lib.stride_tricks.sliding_window_view(input, (3, 3, input.shape[2]), axis=(0, 1, 2)).reshape(h, w, 3*3*input.shape[2])
    kernels_flat = kernels.reshape(num_kernels, 3*3*input.shape[2])
    output = patches @ kernels_flat.T

    return output.reshape(h, w, num_kernels)

def max_pool(input):
    h, w, c = input.shape
    reshaped = input.reshape(h//2, 2, w//2, 2, c)
    return np.max(reshaped, axis=(1, 3))

def dense(input, w, b):
    return input @ w + b

def relu(input):
    return np.maximum(0, input)

def softmax(input):
    logits = input - np.max(input, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def padding(input):
    return np.pad(input, pad_width=((1,1), (1,1), (0,0)), mode='constant', constant_values=0)

def flatten(input):
    return input.flatten()

#BackProp Functions
def max_pool_backward(input, grad_output):
        h, w, c = input.shape
        grad_input = np.zeros(input.shape)

        patches = input.reshape(h//2, 2, w//2, 2, c)
        patch_values = patches.reshape(h//2, w//2, 4, c)
        max_indices = np.argmax(patch_values, axis=2)

        i_coords, j_coords, k_coords = np.meshgrid(
            np.arange(h//2), np.arange(w//2), np.arange(c), indexing='ij'
        )

        row_positions = max_indices // 2
        col_positions = max_indices % 2

        actual_rows = i_coords * 2 + row_positions
        actual_cols = j_coords * 2 + col_positions

        grad_input[actual_rows, actual_cols, k_coords] = grad_output

        return grad_input
    
def conv_backward(input, grad_output, kernels):
    h, w = input.shape[0] - 2, input.shape[1] - 2
    patches = np.lib.stride_tricks.sliding_window_view(input, (3, 3, input.shape[2]), axis=(0, 1, 2)).reshape(h, w, 3*3*input.shape[2])
    grad_flat = grad_output.reshape(h*w, grad_output.shape[2])

def conv_kernels_backward(input, grad_output, kernel_shape):
    h, w = input.shape[0] - 2, input.shape[1] - 2
    patches = np.lib.stride_tricks.sliding_window_view(input, (3, 3, input.shape[2]), axis=(0, 1, 2)).reshape(h, w, 3*3*input.shape[2])
    grad_flat = grad_output.reshape(h*w, grad_output.shape[2])
    grads = (patches.reshape(h*w, -1).T @ grad_flat).T

    return grads.reshape(kernel_shape)

def conv_input_backward(grad_output, kernels, input_shape):
    grad_input = np.zeros(input_shape)
    h, w = grad_output.shape[0], grad_output.shape[1]
    num_kernels = kernels.shape[0]

    flipped_kernels = np.flip(kernels, axis=(1, 2))
    
    for k in range(num_kernels):
        kernel = flipped_kernels[k] 
        grad_k = grad_output[:, :, k:k+1] 
        
        for c in range(kernel.shape[2]):
            grad_input[:, :, c] += signal.convolve2d(
                grad_k[:, :, 0], kernel[:, :, c], mode='full'
            )
    
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

    patches_reshaped = patches.reshape(batch_size, 3*3*c, (h-2)*(w-2))
    grads = np.einsum('bci,bjn->bcn', patches_reshaped, grad_flat)
    
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

def batch_conv_input_backward(grad_output_batch, kernels, input_shape):
    batch_size = grad_output_batch.shape[0]
    grad_input_batch = np.zeros((batch_size,) + input_shape[1:])
    
    flipped_kernels = np.flip(kernels, axis=(1, 2))
    
    for k in range(kernels.shape[0]):
        kernel = flipped_kernels[k]
        grad_k = grad_output_batch[:, :, :, k]
        
        for c in range(kernel.shape[2]):
            for b in range(batch_size):
                grad_input_batch[b, :, :, c] += signal.convolve2d(
                    grad_k[b], kernel[:, :, c], mode='full'
                )
    
    return grad_input_batch

def batch_max_pool_backward(input_batch, grad_output_batch):
    batch_size, h, w, c = input_batch.shape
    grad_input_batch = np.zeros_like(input_batch)
    
    reshaped_input = input_batch.reshape(batch_size, h//2, 2, w//2, 2, c)
    patch_values = reshaped_input.reshape(batch_size, h//2, w//2, 4, c)
    max_indices = np.argmax(patch_values, axis=3)
    
    i_coords, j_coords, k_coords = np.meshgrid(
        np.arange(h//2), np.arange(w//2), np.arange(c), indexing='ij'
    )
    
    i_coords = np.broadcast_to(i_coords, (batch_size, h//2, w//2, c))
    j_coords = np.broadcast_to(j_coords, (batch_size, h//2, w//2, c))
    k_coords = np.broadcast_to(k_coords, (batch_size, h//2, w//2, c))
    
    row_positions = max_indices // 2
    col_positions = max_indices % 2
    
    actual_rows = i_coords * 2 + row_positions
    actual_cols = j_coords * 2 + col_positions
    
    batch_indices = np.arange(batch_size)[:, np.newaxis, np.newaxis, np.newaxis]
    grad_input_batch[batch_indices, actual_rows, actual_cols, k_coords] = grad_output_batch
    
    return grad_input_batch

def conv_scipy(input, kernels):
    num_kernels = kernels.shape[0]
    h, w = input.shape[0] - 2, input.shape[1] - 2
    output = np.zeros((h, w, num_kernels))
    
    for k in range(num_kernels):
        kernel = kernels[k]
        for c in range(kernel.shape[2]):
            output[:, :, k] += signal.correlate2d(
                input[:, :, c], kernel[:, :, c], mode='valid'
            )
    
    return output

def batch_conv_scipy(input_batch, kernels):
    batch_size, h, w, c = input_batch.shape
    num_kernels = kernels.shape[0]
    output = np.zeros((batch_size, h-2, w-2, num_kernels))
    
    for b in range(batch_size):
        for k in range(num_kernels):
            kernel = kernels[k]
            for ch in range(c):
                output[b, :, :, k] += signal.correlate2d(
                    input_batch[b, :, :, ch], kernel[:, :, ch], mode='valid'
                )
    
    return output

def conv_kernels_backward_scipy(input, grad_output, kernel_shape):
    h, w = input.shape[0] - 2, input.shape[1] - 2
    num_kernels = grad_output.shape[2]
    num_channels = input.shape[2]
    
    grads = np.zeros(kernel_shape)
    
    for k in range(num_kernels):
        for c in range(num_channels):
            grads[k, :, :, c] = signal.correlate2d(
                input[:, :, c], grad_output[:, :, k], mode='valid'
            )
    
    return grads

def batch_conv_kernels_backward_scipy(input_batch, grad_output_batch, kernel_shape):
    batch_size, h, w, c = input_batch.shape
    num_kernels = grad_output_batch.shape[3]
    
    grads = np.zeros(kernel_shape)
    
    for b in range(batch_size):
        for k in range(num_kernels):
            for ch in range(c):
                grads[k, :, :, ch] += signal.correlate2d(
                    input_batch[b, :, :, ch], grad_output_batch[b, :, :, k], mode='valid'
                )
    
    return grads / batch_size

def optimize_dtypes():
    return np.float32

def batch_conv_optimized(input_batch, kernels):
    batch_size, h, w, c = input_batch.shape
    num_kernels = kernels.shape[0]
    
    input_batch = input_batch.astype(np.float32)
    kernels = kernels.astype(np.float32)
    
    patches = np.lib.stride_tricks.sliding_window_view(
        input_batch, (3, 3, c), axis=(1, 2, 3)
    ).reshape(batch_size, h-2, w-2, 3*3*c)
    
    patches_reshaped = patches.reshape(batch_size * (h-2) * (w-2), 3*3*c)
    kernels_flat = kernels.reshape(num_kernels, 3*3*c)
    
    output_flat = patches_reshaped @ kernels_flat.T
    
    return output_flat.reshape(batch_size, h-2, w-2, num_kernels)

def batch_dense_optimized(input_batch, w, b):
    if not input_batch.flags['C_CONTIGUOUS']:
        input_batch = np.ascontiguousarray(input_batch, dtype=np.float32)
    else:
        input_batch = input_batch.astype(np.float32)
    
    w = w.astype(np.float32)
    b = b.astype(np.float32)
    
    return input_batch @ w + b
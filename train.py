import numpy as np
from model import CNN
from data_loader import load_cifar10_data

def train_single_sample():
    print("TESTING")

    x_train, y_train, x_test, y_test = load_cifar10_data()
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

def test_network_forward():
    """Test forward pass and show network architecture."""
    x_train, y_train, x_test, y_test = load_cifar10_data()
    cnn = CNN()
    
    sample_x = x_train[0]
    predictions = cnn.forward(sample_x)
    
    print(f"Input shape: {sample_x.shape}")
    print(f"Network layer shapes:")
    print(f"  Conv1 output: {cnn.conv1.shape}")
    print(f"  Pool1 output: {cnn.pool1.shape}")
    print(f"  Conv2 output: {cnn.conv2.shape}")
    print(f"  Pool2 output: {cnn.pool2.shape}")
    print(f"  Flattened: {cnn.flattened.shape}")
    print(f"  Dense1 output: {cnn.dense_relu.shape}")
    print(f"  Final output: {predictions.shape}")

def train_multiple_samples(num_samples=100, epochs=5):
    print("=" * 50)
    print(f"TRAINING ON {num_samples} SAMPLES FOR {epochs} EPOCHS")
    print("=" * 50)
    
    x_train, y_train, x_test, y_test = load_cifar10_data()
    cnn = CNN()
    
    learning_rate = 0.001
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for i in range(num_samples):
            predictions = cnn.forward(x_train[i])
            loss = cnn.compute_loss(predictions, y_train[i])
            cnn.backprop(y_train[i])
            cnn.update_weights(learning_rate)
            
            total_loss += loss
            if np.argmax(predictions) == np.argmax(y_train[i]):
                correct += 1
        
        accuracy = correct / num_samples * 100
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1f}%")

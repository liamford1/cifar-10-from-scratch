import numpy as np
from model import CNN, BatchCNN
from data_loader import load_cifar10_data
from layers import batch_conv, batch_max_pool, batch_relu, batch_padding, batch_flatten

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

def test_shapes():
    """Test if all layer dimensions match up correctly."""
    x_train, y_train, _, _ = load_cifar10_data()
    cnn = CNN()
    
    x = x_train[0]
    print(f"Input: {x.shape}")
    
    pred = cnn.forward(x)
    print(f"After pool2: {cnn.pool2.shape}")
    print(f"After flatten: {cnn.flattened.shape}")
    print(f"Weights1 expects: {cnn.weights1.shape[0]} inputs")
    print(f"Match: {cnn.flattened.shape[0] == cnn.weights1.shape[0]}")

def quick_gradient_check():
    """Quick test to see if gradients are working."""
    x_train, y_train, _, _ = load_cifar10_data()
    cnn = CNN()
    
    x, y = x_train[0], y_train[0]
    
    # Get initial prediction and loss
    pred1 = cnn.forward(x)
    loss1 = cnn.compute_loss(pred1, y)
    
    # Compute gradients
    cnn.backprop(y)
    
    # Update weights with small step
    cnn.update_weights(0.001)
    
    # Check if loss decreased
    pred2 = cnn.forward(x)
    loss2 = cnn.compute_loss(pred2, y)
    
    print(f"Loss before: {loss1:.6f}")
    print(f"Loss after:  {loss2:.6f}")
    print(f"Improved: {loss2 < loss1}")
    
    return loss2 < loss1

def test_batch_training():
    """Test the BatchCNN with multiple samples."""
    print("=" * 50)
    print("TESTING BATCH TRAINING")
    print("=" * 50)
    
    x_train, y_train, _, _ = load_cifar10_data()
    batch_cnn = BatchCNN()
    
    batch_size = 100
    learning_rate = 0.01  # Reduced learning rate
    
    # Get a batch of samples
    batch_x = x_train[:batch_size]
    batch_y = y_train[:batch_size]
    
    print(f"Training on batch of {batch_size} samples")
    print(f"Batch input shape: {batch_x.shape}")
    print(f"Batch labels shape: {batch_y.shape}")
    
    # Initial predictions
    initial_preds = batch_cnn.forward(batch_x)
    initial_loss = batch_cnn.compute_loss(initial_preds, batch_y)
    initial_accuracy = np.mean(np.argmax(initial_preds, axis=1) == np.argmax(batch_y, axis=1)) * 100
    
    print(f"\nBefore training:")
    print(f"Loss: {initial_loss:.4f}")
    print(f"Accuracy: {initial_accuracy:.1f}%")
    
    # Debug: Show predictions vs true labels
    print(f"True labels: {np.argmax(batch_y, axis=1)}")
    print(f"Predictions: {np.argmax(initial_preds, axis=1)}")
    print(f"Prediction confidences: {np.max(initial_preds, axis=1)}")
    
    # Train for a few epochs
    for epoch in range(30):  # More epochs with smaller learning rate
        preds = batch_cnn.forward(batch_x)
        batch_cnn.backprop(batch_y)
        batch_cnn.update_weights(learning_rate)
        
        # Compute new predictions with updated weights for accuracy
        updated_preds = batch_cnn.forward(batch_x)
        
        loss = batch_cnn.compute_loss(preds, batch_y)
        accuracy = np.mean(np.argmax(updated_preds, axis=1) == np.argmax(batch_y, axis=1)) * 100
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.1f}%")
        
        # Debug: Show predictions after epoch
        if epoch == 0 or epoch == 4:  # Show first and last epoch
            print(f"  Predictions: {np.argmax(updated_preds, axis=1)}")
            print(f"  Confidences: {np.max(updated_preds, axis=1)}")
    
    print(f"\nImprovement:")
    print(f"Loss: {initial_loss:.4f} → {loss:.4f}")
    print(f"Accuracy: {initial_accuracy:.1f}% → {accuracy:.1f}%")

def debug_batch_shapes():
    """Debug the shapes in batch processing"""
    print("DEBUGGING BATCH SHAPES")
    
    x_train, y_train, _, _ = load_cifar10_data()
    batch_cnn = BatchCNN()
    
    batch_x = x_train[:2]  # Just 2 samples for debugging
    
    print(f"Input batch: {batch_x.shape}")
    
    # Step by step through forward pass
    padded = batch_padding(batch_x)
    print(f"After padding: {padded.shape}")
    
    conv1 = batch_conv(padded, batch_cnn.conv1_kernels)
    print(f"After conv1: {conv1.shape}")
    
    relu1 = batch_relu(conv1)
    pool1 = batch_max_pool(relu1)
    print(f"After pool1: {pool1.shape}")
    
    second_padding = batch_padding(pool1)
    print(f"After second padding: {second_padding.shape}")
    
    conv2 = batch_conv(second_padding, batch_cnn.conv2_kernels)
    print(f"After conv2: {conv2.shape}")
    
    relu2 = batch_relu(conv2)
    pool2 = batch_max_pool(relu2)
    print(f"After pool2: {pool2.shape}")
    
    flattened = batch_flatten(pool2)
    print(f"After flatten: {flattened.shape}")
    print(f"Weights1 shape: {batch_cnn.weights1.shape}")

def train_and_evaluate(learning_rate, batch_size, epochs):
    """Train model with given hyperparameters and return final accuracy"""
    x_train, y_train, _, _ = load_cifar10_data()
    batch_cnn = BatchCNN()
    
    # Get batch of samples
    batch_x = x_train[:batch_size]
    batch_y = y_train[:batch_size]
    
    # Train
    for epoch in range(epochs):
        preds = batch_cnn.forward(batch_x)
        batch_cnn.backprop(batch_y)
        batch_cnn.update_weights(learning_rate)
    
    # Evaluate
    final_preds = batch_cnn.forward(batch_x)
    accuracy = np.mean(np.argmax(final_preds, axis=1) == np.argmax(batch_y, axis=1)) * 100
    return accuracy

def hyperparameter_search():
    print("=== HYPERPARAMETER SEARCH ===")
    
    # Test learning rates
    print("\nTesting Learning Rates:")
    for lr in [0.001, 0.01, 0.1]:
        acc = train_and_evaluate(lr, 64, 30)
        print(f"LR {lr}: {acc:.1f}%")
    
    # Test batch sizes with best LR
    print("\nTesting Batch Sizes:")
    for batch_size in [32, 64, 128]:
        acc = train_and_evaluate(0.01, batch_size, 30)  # Use best LR
        print(f"Batch {batch_size}: {acc:.1f}%")
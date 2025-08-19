import numpy as np
from model import BatchCNN
from data_loader import load_cifar10_data
from config import ModelConfig, TrainingConfig

def train_model(model_config: ModelConfig, training_config: TrainingConfig, 
                max_samples=None, verbose=True):
    """
    Main training function for the CNN model.
    
    Args:
        model_config: Model architecture configuration
        training_config: Training hyperparameters
        max_samples: Maximum number of samples to use (for quick testing)
        verbose: Whether to print training progress
    
    Returns:
        tuple: (trained_model, training_history)
    """
    # Load data
    x_train, y_train, x_test, y_test = load_cifar10_data()
    
    # Limit samples if specified
    if max_samples:
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]
    
    # Split into train/validation
    split_idx = int(len(x_train) * (1 - training_config.validation_split))
    x_train_split = x_train[:split_idx]
    y_train_split = y_train[:split_idx]
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # Initialize model
    model = BatchCNN(model_config)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    if verbose:
        print(f"Training on {len(x_train_split)} samples")
        print(f"Validation on {len(x_val)} samples")
        print(f"Batch size: {training_config.batch_size}")
        print(f"Learning rate: {training_config.learning_rate}")
        print(f"Epochs: {training_config.epochs}")
        print("-" * 50)
    
    # Training loop
    for epoch in range(training_config.epochs):
        # Training
        train_loss, train_accuracy = train_epoch(
            model, x_train_split, y_train_split, training_config
        )
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, x_val, y_val)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        if verbose:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.1f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.1f}%")
    
    return model, history

def train_epoch(model, x_train, y_train, training_config):
    """Train for one epoch."""
    total_loss = 0
    correct = 0
    num_batches = 0
    
    # Process in batches
    for i in range(0, len(x_train), training_config.batch_size):
        batch_end = min(i + training_config.batch_size, len(x_train))
        batch_x = x_train[i:batch_end]
        batch_y = y_train[i:batch_end]
        
        # Forward pass
        predictions = model.forward(batch_x)
        loss = model.compute_loss(predictions, batch_y)
        
        # Backward pass
        model.backprop(batch_y)
        model.update_weights(training_config.learning_rate)
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(batch_y, axis=1)
        correct += np.sum(pred_classes == true_classes)
        
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = (correct / len(x_train)) * 100
    
    return avg_loss, accuracy

def evaluate_model(model, x_val, y_val):
    """Evaluate model on validation data."""
    total_loss = 0
    correct = 0
    num_batches = 0
    
    for i in range(0, len(x_val), 64):  # Use batch size 64 for evaluation
        batch_end = min(i + 64, len(x_val))
        batch_x = x_val[i:batch_end]
        batch_y = y_val[i:batch_end]
        
        predictions = model.forward(batch_x)
        loss = model.compute_loss(predictions, batch_y)
        
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(batch_y, axis=1)
        correct += np.sum(pred_classes == true_classes)
        
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = (correct / len(x_val)) * 100
    
    return avg_loss, accuracy

def test_model(model, x_test, y_test):
    """Test model on test data."""
    loss, accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.1f}%")
    return loss, accuracy

# Quick training function for experimentation
def quick_train(learning_rate=0.01, batch_size=64, epochs=10, max_samples=1000):
    """Quick training function for testing different configurations."""
    model_config = ModelConfig()
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )
    
    model, history = train_model(
        model_config, training_config, 
        max_samples=max_samples, verbose=True
    )
    
    return model, history
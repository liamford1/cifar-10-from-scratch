"""
CIFAR-10 CNN from Scratch - Main Demo
Shows the complete CNN training from scratch using pure NumPy
"""

from train import train_model, TrainingConfig, quick_train, test_model
from config import ModelConfig
from data_loader import load_cifar10_data

def main():
    print("=" * 60)
    print("CIFAR-10 CNN FROM SCRATCH")
    print("Complete implementation using pure NumPy")
    print("=" * 60)
    
    # Load data info
    x_train, y_train, x_test, y_test = load_cifar10_data()
    print(f"Dataset loaded: {x_train.shape[0]} training samples")
    print(f"Image shape: {x_train.shape[1:]}")
    print()
    
    # Quick training demo
    print("Running quick training demo...")
    model, history = quick_train(
        learning_rate=0.01,
        batch_size=64,
        epochs=5,
        max_samples=1000  # Use subset for quick demo
    )
    
    print("\n" + "=" * 60)
    print("Training complete! Testing on test set...")
    print("=" * 60)
    
    # Test on test set
    test_model(model, x_test, y_test)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
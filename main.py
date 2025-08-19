"""
CIFAR-10 CNN from Scratch - Main Demo
Shows the complete CNN training from scratch using pure NumPy
"""

from train import train_single_sample, test_network_forward, test_shapes, quick_gradient_check, test_batch_training, debug_batch_shapes, hyperparameter_search
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
    
    # Test shapes first
    print("Testing layer dimensions...")
    test_shapes()
    print()
    
    # Quick gradient check
    print("Testing gradient computation...")
    quick_gradient_check()
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    test_network_forward()
    print()
    
    # Demonstrate learning
    print("Demonstrating learning on single sample...")
    train_single_sample()
    
    print("\n" + "=" * 60)
    print("Demo complete! Check train.py for more training options.")
    print("=" * 60)

    # Test batch training
    print("\nTesting batch processing...")
    test_batch_training()

    debug_batch_shapes()

    hyperparameter_search()

if __name__ == "__main__":
    main()
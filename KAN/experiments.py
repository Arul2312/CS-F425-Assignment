import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from .models import KAN
from .training import train_regression_model

def load_and_normalize_data(csv_path, test_size=0.2):
    data = np.loadtxt(csv_path)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Normalize X between -1 and 1
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = -1 + 2 * (X - X_min) / (X_max - X_min + 1e-8)

    # Normalize y between -1 and 1
    y_min = y.min()
    y_max = y.max()
    y_normalized = -1 + 2 * (y - y_min) / (y_max - y_min + 1e-8)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_normalized)
    y_tensor = torch.FloatTensor(y_normalized.reshape(-1, 1))
    
    # Create train/test split
    num_samples = len(X)
    num_train = int((1 - test_size) * num_samples)
    indices = torch.randperm(num_samples)
    
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    return (
        X_tensor[train_indices], y_tensor[train_indices],
        X_tensor[test_indices], y_tensor[test_indices],
        X.shape[1]
    )

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size)
    )

def create_mlp_model(input_dim, output_dim, hidden_dims=[4, 4, 4]):
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(nn.ReLU())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

def compare_models(csv_path, test_size=0.2, batch_size=32, epochs=50, lr=0.001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    X_train, y_train, X_test, y_test, input_dim = load_and_normalize_data(csv_path, test_size)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size)
    
    print(f"Total samples: {len(X_train) + len(X_test)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Input features: {input_dim}")
    
    # Create models
    output_dim = 1
    fixed_kan = KAN([input_dim, 2, 3, 1, output_dim], num_control_points=5, degree=4,
                   sparsity_lambda=0.001, device=device)
    
    unpruned_kan = KAN([input_dim, 3, 3, 3, 3, output_dim], num_control_points=5, degree=4,
                      sparsity_lambda=0.001, device=device)
    
    mlp = create_mlp_model(input_dim, output_dim).to(device)
    
    # Train models
    print("\nTraining Fixed KAN...")
    fixed_kan_train_rmse, fixed_kan_test_rmse = train_regression_model(
        fixed_kan, train_loader, test_loader, epochs=epochs, lr=lr, verbose=verbose)
    
    print("\nTraining Unpruned KAN...")
    unpruned_kan_train_rmse, unpruned_kan_test_rmse = train_regression_model(
        unpruned_kan, train_loader, test_loader, epochs=epochs, lr=lr, verbose=verbose)
    
    print("\nPruning KAN...")
    pruned_kan = unpruned_kan.prune(threshold=0.25, fine_tune=True, train_loader=train_loader, epochs=5)
    print(f"Original KAN architecture: {unpruned_kan.get_architecture()}")
    print(f"Pruned KAN architecture: {pruned_kan.get_architecture()}")
    
    print("\nTraining Pruned KAN...")
    pruned_kan_train_rmse, pruned_kan_test_rmse = train_regression_model(
        pruned_kan, train_loader, test_loader, epochs=epochs, lr=lr, verbose=verbose)
    
    print("\nTraining MLP...")
    mlp_train_rmse, mlp_test_rmse = train_regression_model(
        mlp, train_loader, test_loader, epochs=epochs, lr=lr, verbose=verbose)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(fixed_kan_train_rmse, label='Fixed KAN')
    plt.plot(unpruned_kan_train_rmse, label='Unpruned KAN')
    plt.plot(pruned_kan_train_rmse, label='Pruned KAN')
    plt.plot(mlp_train_rmse, label='MLP')
    plt.xlabel('Epoch')
    plt.ylabel('Training RMSE')
    plt.title('Training RMSE Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(fixed_kan_test_rmse, label='Fixed KAN')
    plt.plot(unpruned_kan_test_rmse, label='Unpruned KAN')
    plt.plot(pruned_kan_test_rmse, label='Pruned KAN')
    plt.plot(mlp_test_rmse, label='MLP')
    plt.xlabel('Epoch')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = 'model_comparison.png'
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()
    
    print("\nFinal Test RMSE Values:")
    print(f"Fixed KAN: {fixed_kan_test_rmse[-1]:.6f}")
    print(f"Unpruned KAN: {unpruned_kan_test_rmse[-1]:.6f}")
    print(f"Pruned KAN: {pruned_kan_test_rmse[-1]:.6f}")
    print(f"MLP: {mlp_test_rmse[-1]:.6f}")
    
    return {
        'fixed_kan': fixed_kan,
        'unpruned_kan': unpruned_kan,
        'pruned_kan': pruned_kan,
        'mlp': mlp,
        'metrics': {
            'train_rmse': {
                'fixed_kan': fixed_kan_train_rmse,
                'unpruned_kan': unpruned_kan_train_rmse,
                'pruned_kan': pruned_kan_train_rmse,
                'mlp': mlp_train_rmse
            },
            'test_rmse': {
                'fixed_kan': fixed_kan_test_rmse,
                'unpruned_kan': unpruned_kan_test_rmse,
                'pruned_kan': pruned_kan_test_rmse,
                'mlp': mlp_test_rmse
            }
        }
    }
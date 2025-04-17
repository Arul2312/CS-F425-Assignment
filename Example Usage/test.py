import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pandas as pd

class BSplineActivation(nn.Module):
    def __init__(self, num_control_points=5, degree=4, init_scale=0.1, grid_range=(-1, 1), device="cuda"):
        super().__init__()
        self.degree = degree
        self.num_control_points = num_control_points
        self.grid_range = grid_range
        self.device = device
        self.control_points = nn.Parameter(torch.randn(num_control_points, device=device) * init_scale)
        knots_np = np.linspace(grid_range[0] - degree * 1e-1, grid_range[1] + degree * 1e-1, num_control_points + degree + 1)
        self.register_buffer('knots', torch.tensor(knots_np, dtype=torch.float32, device=device))
        self.base_func = lambda x: x * torch.sigmoid(x)
        self.base_weight = nn.Parameter(torch.randn(1, device=device))
        self.spline_weight = nn.Parameter(torch.randn(1, device=device))
        self.set_grid_points(3)  # Initialize with G=3

    def set_grid_points(self, G):
        """Update grid points and precompute basis values."""
        self.grid_points = torch.linspace(self.grid_range[0], self.grid_range[1], G, device=self.device)
        self.basis_values = self._precompute_basis(self.grid_points, self.knots, self.degree)

    def _precompute_basis(self, grid_points, knots, degree):
        basis_values = torch.zeros((len(grid_points), self.num_control_points), device=self.device)
        for i in range(self.num_control_points):
            center = (knots[i+degree//2] + knots[i+degree//2+1]) / 2
            width = (knots[i+degree+1] - knots[i]) / 2
            basis_values[:, i] = torch.exp(-((grid_points - center) / width)**2)
        basis_sum = basis_values.sum(dim=1, keepdim=True)
        return basis_values / (basis_sum + 1e-6)

    def forward(self, x):
        x_clipped = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        norm_x = (x_clipped - self.grid_range[0]) / (self.grid_range[1] - self.grid_range[0])
        grid_idx = norm_x * (len(self.grid_points) - 1)
        idx_low = torch.clamp(torch.floor(grid_idx).long(), 0, len(self.grid_points) - 1)
        idx_high = torch.clamp(torch.ceil(grid_idx).long(), 0, len(self.grid_points) - 1)
        frac = grid_idx - idx_low
        basis_low = self.basis_values[idx_low]
        basis_high = self.basis_values[idx_high]
        basis = basis_low * (1 - frac.unsqueeze(-1)) + basis_high * frac.unsqueeze(-1)
        y_spline = torch.matmul(basis, self.control_points)
        return self.base_weight * self.base_func(x) + self.spline_weight * y_spline


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_control_points=5, degree=3, device="cuda", is_input_or_output=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.is_input_or_output = is_input_or_output
        self.activations = nn.ModuleList([
            BSplineActivation(num_control_points, degree, device=device)
            for _ in range(input_dim * output_dim)
        ])
        self.in_scores = nn.Parameter(torch.ones(input_dim, device=device))
        self.out_scores = nn.Parameter(torch.ones(output_dim, device=device))
        self._pruned = False
        self.activation_importance = nn.Parameter(torch.ones(input_dim, output_dim, device=device))

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # Initialize output with correct dimensions
        out = torch.zeros(batch_size, self.output_dim, device=self.device)
        
        for j in range(self.output_dim):
            for i in range(min(self.input_dim, x.shape[1])):
                idx = i * self.output_dim + j
                out[:, j] += self.activation_importance[i, j] * self.activations[idx](x[:, i])
        return out

    def compute_node_scores(self):
        """Compute importance scores for nodes based on activation importance."""
        # Compute input node importance by averaging across all outgoing connections
        input_importance = torch.abs(self.activation_importance).mean(dim=1)
        
        # Compute output node importance by averaging across all incoming connections
        output_importance = torch.abs(self.activation_importance).mean(dim=0)
        
        # Combine with the existing scores for more stable pruning
        input_scores = input_importance * torch.abs(self.in_scores)
        output_scores = output_importance * torch.abs(self.out_scores)
        
        return input_scores, output_scores

    def prune(self, threshold=0.1, fine_tune=False, train_loader=None, epochs=10):
        """
        Enhanced pruning that respects input/output layer preservation and uses better node scoring
        """
        if self._pruned or self.is_input_or_output:
            # Don't prune input/output layers or already pruned layers
            return list(range(self.input_dim)), list(range(self.output_dim))
            
        input_scores, output_scores = self.compute_node_scores()
        
        # Dynamic threshold that adapts to the magnitude of scores
        dynamic_threshold = threshold * max(input_scores.mean().item(), output_scores.mean().item())

        # Create masks for kept nodes
        input_mask = input_scores > dynamic_threshold
        output_mask = output_scores > dynamic_threshold
        
        # Ensure we keep at least one node in each dimension
        if not input_mask.any():
            input_mask[input_scores.argmax()] = True
        if not output_mask.any():
            output_mask[output_scores.argmax()] = True
        
        kept_inputs = torch.where(input_mask)[0].tolist()
        kept_outputs = torch.where(output_mask)[0].tolist()
        
        self._pruned = True
        
        if fine_tune and train_loader:
            self.fine_tune(train_loader, epochs, kept_inputs, kept_outputs)
        
        return kept_inputs, kept_outputs

    def fine_tune(self, train_loader, epochs=10, kept_inputs=None, kept_outputs=None):
        """Fine-tune the layer after pruning with focus on retained nodes."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.train()
        
        if kept_inputs is not None and kept_outputs is not None:
            # Create masks for the kept nodes
            input_mask = torch.zeros(self.input_dim, device=self.device, dtype=torch.bool)
            output_mask = torch.zeros(self.output_dim, device=self.device, dtype=torch.bool)
            
            input_mask[kept_inputs] = True
            output_mask[kept_outputs] = True
            
            # Create a mask for the activations to update
            activation_mask = torch.zeros(self.input_dim, self.output_dim, device=self.device, dtype=torch.bool)
            for i in kept_inputs:
                for j in kept_outputs:
                    activation_mask[i, j] = True
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self(data)
                
                # Ensure output matches target dimensions
                if output.shape != target.shape:
                    output = output[:, :target.shape[1]] if output.shape[1] > target.shape[1] else output
                
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradients only to kept nodes if masks are defined
                if kept_inputs is not None and kept_outputs is not None:
                    with torch.no_grad():
                        # Zero out gradients for pruned connections
                        self.activation_importance.grad[~activation_mask] = 0
                        
                        # Zero out gradients for pruned input/output nodes
                        self.in_scores.grad[~input_mask] = 0
                        self.out_scores.grad[~output_mask] = 0
                
                optimizer.step()


class KAN(nn.Module):
    """Enhanced KAN with improved pruning and architecture search."""
    def __init__(self, layer_dims, num_control_points=5, degree=3, 
                sparsity_lambda=0.01, device="cuda"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.sparsity_lambda = sparsity_lambda
        self.device = device
        self.layer_dims = layer_dims
        self.num_control_points = num_control_points
        self.degree = degree
        self.grid_points = 3  # Start with G=3
        
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layer = KANLayer(in_dim, out_dim, num_control_points, degree, device=device)
            layer.is_input_or_output = (i == 0) or (i == len(layer_dims) - 2)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_grid_points(self, G):
        """Update grid points for all B-Spline activations."""
        for layer in self.layers:
            for activation in layer.activations:
                activation.set_grid_points(G)
        self.grid_points = G

    def regularization_loss(self):
        """Enhanced regularization loss with L1 on activation importance and entropy."""
        l1_loss = 0
        entropy_loss = 0
        
        for layer in self.layers:
            # L1 regularization on control points and activation importance
            for activation in layer.activations:
                l1_loss += torch.mean(torch.abs(activation.control_points))
            
            # Add L1 regularization on activation importance
            l1_loss += torch.mean(torch.abs(layer.activation_importance))
            
            # Entropy regularization on input/output scores
            in_probs = F.softmax(layer.in_scores, dim=0)
            entropy_loss += -torch.sum(in_probs * torch.log(in_probs + 1e-10))
            
            out_probs = F.softmax(layer.out_scores, dim=0)
            entropy_loss += -torch.sum(out_probs * torch.log(out_probs + 1e-10))
        
        return self.sparsity_lambda * (l1_loss + entropy_loss)

    def prune(self, threshold=0.1, fine_tune=False, train_loader=None, epochs=10):
        """
        Enhanced pruning with better architecture discovery
        """
        kept_nodes = []
        # Always keep all input nodes
        kept_nodes.append(list(range(self.layers[0].input_dim)))
        
        for i, layer in enumerate(self.layers):
            # Mark input/output layers
            is_input_or_output = (i == 0) or (i == len(self.layers) - 1)
            
            # Get kept nodes (input/output layers won't be pruned)
            kept_inputs, kept_outputs = layer.prune(
                threshold=threshold if not is_input_or_output else 0,
                train_loader=train_loader if fine_tune and not is_input_or_output else None,
                epochs=epochs if fine_tune and not is_input_or_output else 0
            )
            
            # Filter inputs based on previous layer's kept outputs
            if i > 0:
                kept_inputs = [idx for idx in kept_inputs if idx in kept_nodes[-1]]
            
            kept_nodes.append(kept_outputs)
        
        # Remove empty layers (except input/output)
        pruned_dims = []
        for i, nodes in enumerate(kept_nodes):
            if len(nodes) > 0 or i == 0 or i == len(kept_nodes) - 1:
                pruned_dims.append(len(nodes))
        
        # Create new model with pruned architecture
        pruned_kan = KAN(
            pruned_dims,
            self.num_control_points,
            self.degree,
            self.sparsity_lambda,
            device=self.device
        )
        
        # Set the same grid points as the original model
        pruned_kan.update_grid_points(self.grid_points)
        
        # Copy relevant parameters
        with torch.no_grad():
            for orig_layer, new_layer, (prev_kept, curr_kept) in zip(
                self.layers, pruned_kan.layers, zip(kept_nodes[:-1], kept_nodes[1:])):
                
                # Copy activation parameters for kept connections
                for i_new, i_orig in enumerate(prev_kept):
                    for j_new, j_orig in enumerate(curr_kept):
                        orig_idx = i_orig * orig_layer.output_dim + j_orig
                        new_idx = i_new * new_layer.output_dim + j_new
                        
                        if new_idx < len(new_layer.activations) and orig_idx < len(orig_layer.activations):
                            new_layer.activations[new_idx].load_state_dict(
                                orig_layer.activations[orig_idx].state_dict())
                            
                            # Copy activation importance
                            new_layer.activation_importance[i_new, j_new] = orig_layer.activation_importance[i_orig, j_orig]
                
                # Copy node scores for kept nodes
                for i_new, i_orig in enumerate(prev_kept):
                    new_layer.in_scores[i_new] = orig_layer.in_scores[i_orig]
                
                for j_new, j_orig in enumerate(curr_kept):
                    new_layer.out_scores[j_new] = orig_layer.out_scores[j_orig]
        
        return pruned_kan

    def get_architecture(self):
        """Return the current architecture dimensions."""
        return self.layer_dims


def train_regression_model(model, train_loader, test_loader, epochs=50, optimizer_type='adam',
                         lr=0.001, device='cuda', grid_schedule=None, verbose=True):
    """
    Enhanced training function with LBFGS support and progressive grid refinement.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        optimizer_type: 'adam' or 'lbfgs'
        lr: Learning rate
        device: Device to use for training
        grid_schedule: Dictionary mapping steps to grid points G
        verbose: Whether to print progress
    """
    model = model.to(device)
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(
            model.parameters(), 
            lr=lr,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn="strong_wolfe"
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Use MSE loss for regression
    criterion = nn.MSELoss()
    train_rmse = []
    test_rmse = []
    
    # Set up the grid schedule
    if grid_schedule is None:
        grid_schedule = {0: 3, 200: 5, 400: 10, 600: 20, 800: 50, 1000: 100, 1200: 200}
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", disable=not verbose)
    
    global_step = 0
    
    for epoch in epoch_pbar:
        model.train()
        epoch_losses = []
        
        # Update grid points if needed
        if global_step in grid_schedule and hasattr(model, 'update_grid_points'):
            G = grid_schedule[global_step]
            if verbose:
                print(f"\nUpdating grid points to G={G}")
            model.update_grid_points(G)
        
        # Training with different handling for LBFGS and Adam
        if optimizer_type.lower() == 'lbfgs':
            # LBFGS requires a closure
            def closure():
                optimizer.zero_grad()
                total_loss = 0
                
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    if hasattr(model, 'regularization_loss'):
                        loss = criterion(output, target) + model.regularization_loss()
                    else:
                        loss = criterion(output, target)
                    
                    total_loss += loss.item() * len(data)
                    loss.backward()
                
                return total_loss / len(train_loader.dataset)
            
            # Perform optimization step
            loss = optimizer.step(closure)
            epoch_losses.append(loss)
            
        else:  # Adam or other optimizers
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                
                # For KAN, add regularization loss
                if hasattr(model, 'regularization_loss'):
                    loss = criterion(output, target) + model.regularization_loss()
                else:
                    loss = criterion(output, target)
                    
                loss.backward()
                optimizer.step()
                
                # Store batch loss
                epoch_losses.append(loss.item())
                
                global_step += 1
                
                # Update grid points if needed (inside the batch loop for Adam)
                if global_step in grid_schedule and hasattr(model, 'update_grid_points'):
                    G = grid_schedule[global_step]
                    if verbose:
                        print(f"\nUpdating grid points to G={G}")
                    model.update_grid_points(G)
        
        # Calculate RMSE for training data
        train_mse = np.mean(epoch_losses)
        current_train_rmse = np.sqrt(train_mse)
        train_rmse.append(current_train_rmse)
        
        # Evaluate on test set
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss = criterion(output, target).item()
                test_losses.append(test_loss)
        
        # Calculate RMSE for test data
        test_mse = np.mean(test_losses)
        current_test_rmse = np.sqrt(test_mse)
        test_rmse.append(current_test_rmse)
        
        # Update epoch progress bar with current metrics
        epoch_pbar.set_postfix({
            "Train RMSE": f"{current_train_rmse:.4f}", 
            "Test RMSE": f"{current_test_rmse:.4f}",
            "G": f"{model.grid_points if hasattr(model, 'grid_points') else 'N/A'}"
        })
        
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f'Epoch {epoch+1}/{epochs}, Train RMSE: {train_rmse[-1]:.4f}, '
                  f'Test RMSE: {test_rmse[-1]:.4f}, '
                  f'Grid points: {model.grid_points if hasattr(model, "grid_points") else "N/A"}')
    
    return train_rmse, test_rmse


def compare_models_for_equation(csv_path, test_size=0.2, batch_size=32, epochs=50, lr=0.001, verbose=True):
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ------ Data Loading Section ------
    # Load data from CSV file
    print(f"Loading data from {csv_path}...")
    data = np.loadtxt(csv_path)
    
    # Split into features (X) and target (y)
    X = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Last column
    # Normalize X between -1 and 1
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = -1 + 2 * (X - X_min) / (X_max - X_min + 1e-8)  # Add small constant to avoid division by zero

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
    
    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_test = X_tensor[test_indices]
    y_test = y_tensor[test_indices]
    
    # Create DataLoader objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Print dataset info
    print(f"Total samples: {num_samples}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Testing samples: {len(test_indices)}")
    print(f"Input features: {X.shape[1]}")
    
    # ------ Model Training Function ------
    # def train_regression_model(model, train_loader, test_loader, epochs=50, lr=0.001, verbose=True):
    #     model = model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    #     # Use MSE loss for regression
    #     criterion = nn.MSELoss()
        
    #     train_rmse = []
    #     test_rmse = []
        
    #     for epoch in range(epochs):
    #         model.train()
    #         epoch_losses = []
            
    #         for batch_idx, (data, target) in enumerate(train_loader):
    #             data, target = data.to(device), target.to(device)
    #             optimizer.zero_grad()
                
    #             output = model(data)
                
    #             # For KAN, add regularization loss
    #             if isinstance(model, KAN):
    #                 loss = criterion(output, target) + model.regularization_loss()
    #             else:
    #                 loss = criterion(output, target)
                
    #             loss.backward()
    #             optimizer.step()
                
    #             # Store batch loss
    #             epoch_losses.append(loss.item())
            
    #         # Calculate RMSE for training data
    #         train_mse = np.mean(epoch_losses)
    #         train_rmse.append(np.sqrt(train_mse))
            
    #         # Evaluate on test set
    #         model.eval()
    #         test_losses = []
            
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(device), target.to(device)
    #                 output = model(data)
    #                 test_loss = criterion(output, target).item()
    #                 test_losses.append(test_loss)
            
    #         # Calculate RMSE for test data
    #         test_mse = np.mean(test_losses)
    #         test_rmse.append(np.sqrt(test_mse))
            
    #         if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
    #             print(f'Epoch {epoch+1}/{epochs}, Train RMSE: {train_rmse[-1]:.4f}, '
    #                   f'Test RMSE: {test_rmse[-1]:.4f}')
        
    #     return train_rmse, test_rmse
    
    # ------ Model Creation and Comparison ------
    # Get input and output dimensions from data
    input_dim = X.shape[1]
    output_dim = 1  # For regression
    
    # Create models
    fixed_kan = KAN([input_dim, 2, 3, 1, output_dim], num_control_points=5, degree=4,
                   sparsity_lambda=0.001, device=device)
    
    unpruned_kan = KAN([input_dim, 3, 3, 3, 3, output_dim], num_control_points=5, degree=4,
                      sparsity_lambda=0.001, device=device)
    
    mlp = nn.Sequential(
        nn.Linear(input_dim, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, output_dim)
    ).to(device)
    
    # Train the models
    print("\nTraining Fixed KAN...")
    fixed_kan_train_rmse, fixed_kan_test_rmse = train_regression_model(
        fixed_kan, train_loader, test_loader, epochs=epochs, lr=lr, verbose=verbose)
    
    print("\nTraining Unpruned KAN...")
    unpruned_kan_train_rmse, unpruned_kan_test_rmse = train_regression_model(
        unpruned_kan, train_loader, test_loader, epochs=epochs, lr=lr, verbose=verbose)
    
    # Prune and train KAN
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
    
    # ------ Plot Results ------
    plt.figure(figsize=(15, 10))
    
    # Training RMSE plot
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
    
    # Test RMSE plot
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
    filename = 'III.9.52.png'
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()
    
    # Print final RMSE values
    print("\nFinal Test RMSE Values:")
    print(f"Fixed KAN: {fixed_kan_test_rmse[-1]:.6f}")
    print(f"Unpruned KAN: {unpruned_kan_test_rmse[-1]:.6f}")
    print(f"Pruned KAN: {pruned_kan_test_rmse[-1]:.6f}")
    print(f"MLP: {mlp_test_rmse[-1]:.6f}")
    
    # Return trained models and metrics for further analysis if needed
if __name__ == "__main__":
    csv_path = 'equation_046.csv'  # Replace with your CSV file path
    compare_models_for_equation(csv_path, test_size=0.2, batch_size=32, epochs=25, lr=0.001)
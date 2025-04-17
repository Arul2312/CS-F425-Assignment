import torch
import numpy as np
from tqdm import tqdm

def train_regression_model(model, train_loader, test_loader, epochs=50, optimizer_type='adam',
                         lr=0.001, device='cuda', grid_schedule=None, verbose=True):
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
    
    criterion = nn.MSELoss()
    train_rmse = []
    test_rmse = []
    
    if grid_schedule is None:
        grid_schedule = {0: 3, 200: 5, 400: 10, 600: 20, 800: 50, 1000: 100, 1200: 200}
    
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", disable=not verbose)
    global_step = 0
    
    for epoch in epoch_pbar:
        model.train()
        epoch_losses = []
        
        if global_step in grid_schedule and hasattr(model, 'update_grid_points'):
            G = grid_schedule[global_step]
            if verbose:
                print(f"\nUpdating grid points to G={G}")
            model.update_grid_points(G)
        
        if optimizer_type.lower() == 'lbfgs':
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
            
            loss = optimizer.step(closure)
            epoch_losses.append(loss)
            
        else:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                
                if hasattr(model, 'regularization_loss'):
                    loss = criterion(output, target) + model.regularization_loss()
                else:
                    loss = criterion(output, target)
                    
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                global_step += 1
                
                if global_step in grid_schedule and hasattr(model, 'update_grid_points'):
                    G = grid_schedule[global_step]
                    if verbose:
                        print(f"\nUpdating grid points to G={G}")
                    model.update_grid_points(G)
        
        train_mse = np.mean(epoch_losses)
        current_train_rmse = np.sqrt(train_mse)
        train_rmse.append(current_train_rmse)
        
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss = criterion(output, target).item()
                test_losses.append(test_loss)
        
        test_mse = np.mean(test_losses)
        current_test_rmse = np.sqrt(test_mse)
        test_rmse.append(current_test_rmse)
        
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
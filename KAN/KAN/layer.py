import torch
import torch.nn as nn
import torch.nn.functional as F
from .splines import BSplineActivation

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
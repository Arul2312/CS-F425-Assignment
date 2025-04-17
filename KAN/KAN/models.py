import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import KANLayer

class KAN(nn.Module):
    def __init__(self, layer_dims, num_control_points=5, degree=3, 
                sparsity_lambda=0.01, device="cuda"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.sparsity_lambda = sparsity_lambda
        self.device = device
        self.layer_dims = layer_dims
        self.num_control_points = num_control_points
        self.degree = degree
        self.grid_points = 3
        
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layer = KANLayer(in_dim, out_dim, num_control_points, degree, device=device)
            layer.is_input_or_output = (i == 0) or (i == len(layer_dims) - 2)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_grid_points(self, G):
        for layer in self.layers:
            for activation in layer.activations:
                activation.set_grid_points(G)
        self.grid_points = G

    def regularization_loss(self):
        l1_loss = 0
        entropy_loss = 0
        
        for layer in self.layers:
            for activation in layer.activations:
                l1_loss += torch.mean(torch.abs(activation.control_points))
            l1_loss += torch.mean(torch.abs(layer.activation_importance))
            
            in_probs = F.softmax(layer.in_scores, dim=0)
            entropy_loss += -torch.sum(in_probs * torch.log(in_probs + 1e-10))
            
            out_probs = F.softmax(layer.out_scores, dim=0)
            entropy_loss += -torch.sum(out_probs * torch.log(out_probs + 1e-10))
        
        return self.sparsity_lambda * (l1_loss + entropy_loss)

    def prune(self, threshold=0.1, fine_tune=False, train_loader=None, epochs=10):
        kept_nodes = []
        kept_nodes.append(list(range(self.layers[0].input_dim)))
        
        for i, layer in enumerate(self.layers):
            is_input_or_output = (i == 0) or (i == len(self.layers) - 1)
            kept_inputs, kept_outputs = layer.prune(
                threshold=threshold if not is_input_or_output else 0,
                train_loader=train_loader if fine_tune and not is_input_or_output else None,
                epochs=epochs if fine_tune and not is_input_or_output else 0
            )
            
            if i > 0:
                kept_inputs = [idx for idx in kept_inputs if idx in kept_nodes[-1]]
            
            kept_nodes.append(kept_outputs)
        
        pruned_dims = []
        for i, nodes in enumerate(kept_nodes):
            if len(nodes) > 0 or i == 0 or i == len(kept_nodes) - 1:
                pruned_dims.append(len(nodes))
        
        pruned_kan = KAN(
            pruned_dims,
            self.num_control_points,
            self.degree,
            self.sparsity_lambda,
            device=self.device
        )
        pruned_kan.update_grid_points(self.grid_points)
        
        with torch.no_grad():
            for orig_layer, new_layer, (prev_kept, curr_kept) in zip(
                self.layers, pruned_kan.layers, zip(kept_nodes[:-1], kept_nodes[1:])):
                
                for i_new, i_orig in enumerate(prev_kept):
                    for j_new, j_orig in enumerate(curr_kept):
                        orig_idx = i_orig * orig_layer.output_dim + j_orig
                        new_idx = i_new * new_layer.output_dim + j_new
                        
                        if new_idx < len(new_layer.activations) and orig_idx < len(orig_layer.activations):
                            new_layer.activations[new_idx].load_state_dict(
                                orig_layer.activations[orig_idx].state_dict())
                            new_layer.activation_importance[i_new, j_new] = orig_layer.activation_importance[i_orig, j_orig]
                
                for i_new, i_orig in enumerate(prev_kept):
                    new_layer.in_scores[i_new] = orig_layer.in_scores[i_orig]
                
                for j_new, j_orig in enumerate(curr_kept):
                    new_layer.out_scores[j_new] = orig_layer.out_scores[j_orig]
        
        return pruned_kan

    def get_architecture(self):
        return self.layer_dims
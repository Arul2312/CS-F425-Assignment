import torch
import torch.nn as nn
import numpy as np


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

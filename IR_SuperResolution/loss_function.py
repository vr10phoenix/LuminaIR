import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Loss for image restoration"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class EdgeLoss(nn.Module):
    """Forces to generate sharp structural boundaries"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Laplacian kernel to extract high-frequency edges
        k = torch.tensor([[0.5, 1.0, 0.5],
                          [1.0, -6.0, 1.0],
                          [0.5, 1.0, 0.5]]).view(1, 1, 3, 3)
        self.register_buffer('kernel', k)
        self.loss_fn = CharbonnierLoss()

    def forward(self, x, y):
        x_edges = F.conv2d(x, self.kernel, padding=1)
        y_edges = F.conv2d(y, self.kernel, padding=1)
        return self.loss_fn(x_edges, y_edges)

class SuperResolutionLoss(nn.Module):
    """Loss Equation"""
    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        loss_charb = self.charbonnier(pred, target)
        loss_edge = self.edge(pred, target)
        return loss_charb + (self.edge_weight * loss_edge)
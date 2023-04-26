import torch
from torch import nn

class MeanSqrtError(nn.Module):
    def __init__(self):
        super(MeanSqrtError, self).__init__()

    def forward(self, output: torch.tensor, target: torch.tensor) -> float:
        abs_sqrt_error = torch.sqrt(abs(output - target))
        sum_sqrt_error = torch.sum(abs_sqrt_error)
        loss = (sum_sqrt_error / target.size(dim=0))
        
        return loss
import torch.nn as nn
import torch


class trackOutput(nn.Module):
    """
    Takes variable length arguments usually 1 or 2
    Returns the indices of output tensors to be summed
    Written as a helper to accomodate Residual block with Sequential in pyTorch
    """
    def __init__(self, *args):
        super(trackOutput, self).__init__()
        self.tracker = []
        for arg in args:
            self.tracker.append(arg)

    def forward(self):
        return self.tracker



class Concatenate(nn.Module):
    """
    Store indices of tensors to be concatenated
    Indices refer to elements of a ModuleList built from .cfg file
    """
    def __init__(self, i1, i2):
        super(Concatenate, self).__init__()
        self.i1 = i1
        self.i2 = i2

    def forward(self):
        return self.i1, self.i2

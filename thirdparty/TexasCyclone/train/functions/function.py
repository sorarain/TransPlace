import torch.nn as nn

from data.graph import Layout


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, layout: Layout, *args, **kwargs):
        raise NotImplementedError


class MetricFunction:
    def __init__(self):
        pass

    def calculate(self, layout: Layout, *args, **kwargs):
        raise NotImplementedError

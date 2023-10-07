import torch

from data.graph import Layout
from .function import LossFunction, MetricFunction


class HPWLLoss(LossFunction):
    def __init__(self, device):
        super(HPWLLoss, self).__init__()
        self.cal_vector = torch.tensor([[-1], [-1], [1], [1]], dtype=torch.float32).to(device)

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        net_span = layout.net_span
        net_wl = net_span @ self.cal_vector
        return torch.mean(net_wl)


class HPWLMetric(MetricFunction):
    def __init__(self, device):
        super(HPWLMetric, self).__init__()
        self.cal_vector = torch.tensor([[-1], [-1], [1], [1]], dtype=torch.float32).to(device)

    def calculate(self, layout: Layout, *args, **kwargs) -> float:
        net_span = layout.net_span
        net_wl = net_span @ self.cal_vector
        return float(torch.sum(net_wl).cpu().clone().detach().data)

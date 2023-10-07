import torch

from data.graph import Layout
from .function import LossFunction, MetricFunction


class AreaLoss(LossFunction):
    def __init__(self):
        super(AreaLoss, self).__init__()

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        limit = kwargs['limit']
        cell_span = layout.cell_span
        cell_size = layout.cell_size.to(cell_span.device)
        cell_size_opp = cell_size[:, [1, 0]]
        cell_span_excess = torch.relu(torch.cat([
            (torch.tensor(limit[:2], dtype=torch.float32,device=cell_span.device) - cell_span[:, :2]) * cell_size_opp,
            (cell_span[:, 2:] - torch.tensor(limit[2:], dtype=torch.float32,device=cell_span.device)) * cell_size_opp,
        ], dim=-1))
        return torch.mean(cell_span_excess)


class AreaMetric(MetricFunction):
    def __init__(self):
        super(AreaMetric, self).__init__()

    def calculate(self, layout: Layout, *args, **kwargs) -> float:
        limit = kwargs['limit']
        cell_span = layout.cell_span
        cell_size = layout.cell_size.to(cell_span.device)
        cell_size_opp = cell_size[:, [1, 0]]
        cell_span_excess = torch.relu(torch.cat([
            (torch.tensor(limit[:2], dtype=torch.float32,device=cell_span.device) - cell_span[:, :2]) * cell_size_opp,
            (cell_span[:, 2:] - torch.tensor(limit[2:], dtype=torch.float32,device=cell_span.device)) * cell_size_opp,
        ], dim=-1))
        return float(torch.sum(cell_span_excess).cpu().clone().detach().data)

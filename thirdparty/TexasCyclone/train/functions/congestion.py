import numpy as np
import torch
from typing import Tuple, List

from data.graph import Layout
from .function import LossFunction


def greedy_sample_net(net_central_pos: torch.Tensor, span) -> Tuple[List[int], List[int]]:
    net_pos_ = net_central_pos.cpu().detach().numpy()
    sorted_x = np.argsort(net_pos_[:, 0])
    sorted_y = np.argsort(net_pos_[:, 1])
    sample_i, sample_j = [], []
    for sorted_indices in [sorted_x, sorted_y]:
        n = len(sorted_indices)
        for i in range(n):
            for d in range(1, span + 1):
                if i + d < n:
                    sample_i.append(sorted_indices[i])
                    sample_j.append(sorted_indices[i + d])
    return sample_i, sample_j


class SampleNetOverlapLoss(LossFunction):
    def __init__(self, device, span=4):
        super(SampleNetOverlapLoss, self).__init__()
        self.span = span
        self.span2pos_matrix = torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.0],
            [0.0, 0.5],
        ], dtype=torch.float32, device=device)
        self.span2size_matrix = torch.tensor([
            [-1, 0],
            [0, -1],
            [1, 0],
            [0, 1],
        ], dtype=torch.float32, device=device)

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        net_span = layout.net_span
        net_central_pos = net_span @ self.span2pos_matrix.to(net_span.device)
        net_size = net_span @ self.span2size_matrix.to(net_span.device)
        sample_i, sample_j = greedy_sample_net(net_central_pos, self.span)
        if len(sample_i) == 0:
            return torch.tensor(0.)
        sample_net_size_i = net_size[sample_i, :].to(net_span.device)
        sample_net_size_j = net_size[sample_j, :].to(net_span.device)
        sample_net_pos_i = net_central_pos[sample_i, :].to(net_span.device)
        sample_net_pos_j = net_central_pos[sample_j, :].to(net_span.device)
        overlap_x = torch.relu((sample_net_size_i[:, 0] + sample_net_size_j[:, 0]) / 2 -
                               torch.abs(sample_net_pos_i[:, 0] - sample_net_pos_j[:, 0]))
        overlap_y = torch.relu((sample_net_size_i[:, 1] + sample_net_size_j[:, 1]) / 2 -
                               torch.abs(sample_net_pos_i[:, 1] - sample_net_pos_j[:, 1]))
        return torch.mean(torch.sqrt(overlap_x * overlap_y + 1e-3))

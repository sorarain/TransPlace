import numpy as np
import torch
import tqdm
from typing import Tuple, List

from data.graph import Layout
from .function import LossFunction, MetricFunction


def greedy_sample(layout: Layout, span) -> Tuple[List[int], List[int]]:
    cell_pos = layout.cell_pos.cpu().detach().numpy()
    sorted_x = np.argsort(cell_pos[:, 0])
    sorted_y = np.argsort(cell_pos[:, 1])
    sample_i, sample_j = [], []
    for sorted_indices in [sorted_x, sorted_y]:
        n = len(sorted_indices)
        for i in range(n):
            for d in range(1, span + 1):
                if i + d < n:
                    sample_i.append(sorted_indices[i])
                    sample_j.append(sorted_indices[i + d])
    return sample_i, sample_j


def macro_sample(layout: Layout, max_cap) -> Tuple[List[int], List[int]]:
    macro_cell_indices = layout.netlist.terminal_indices
    n_macro = len(macro_cell_indices)
    max_cap = min(max_cap, n_macro)
    sample_i, sample_j = [], []
    for i in macro_cell_indices:
#     for i in tqdm.tqdm(macro_cell_indices):
        for j in macro_cell_indices[:int(max_cap / 2)] + [macro_cell_indices[s] for s in np.random.randint(int(max_cap / 2), n_macro, size=max_cap)]:
            if i != j:
                sample_i.append(i)
                sample_j.append(j)
    return sample_i, sample_j


def calc_overlap_xy(layout: Layout, sample_i: List[int], sample_j: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    cell_size = layout.netlist.graph.nodes['cell'].data['size'].to(layout.cell_pos.device)
    cell_pos = layout.cell_pos
    sample_cell_size_i = cell_size[sample_i, :]
    sample_cell_size_j = cell_size[sample_j, :]
    sample_cell_pos_i = cell_pos[sample_i, :]
    sample_cell_pos_j = cell_pos[sample_j, :]
    overlap_x = torch.relu((sample_cell_size_i[:, 0] + sample_cell_size_j[:, 0]) / 2 -
                           torch.abs(sample_cell_pos_i[:, 0] - sample_cell_pos_j[:, 0]))
    overlap_y = torch.relu((sample_cell_size_i[:, 1] + sample_cell_size_j[:, 1]) / 2 -
                           torch.abs(sample_cell_pos_i[:, 1] - sample_cell_pos_j[:, 1]))
    return overlap_x, overlap_y


class SampleOverlapLoss(LossFunction):
    def __init__(self, span=4):
        super(SampleOverlapLoss, self).__init__()
        self.span = span

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        sample_i, sample_j = greedy_sample(layout, self.span)
        overlap_x, overlap_y = calc_overlap_xy(layout, sample_i, sample_j)
        return torch.mean(torch.sqrt(overlap_x * overlap_y + 1e-3))


class MacroOverlapLoss(LossFunction):
    def __init__(self, max_cap=500):
        super(MacroOverlapLoss, self).__init__()
        self.max_cap = max_cap

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        sample_i, sample_j = macro_sample(layout, self.max_cap)
        if len(sample_i) == 0:
            return torch.tensor(0.)
        overlap_x, overlap_y = calc_overlap_xy(layout, sample_i, sample_j)
        return torch.mean(torch.sqrt(overlap_x * overlap_y + 1e-3))


class OverlapMetric(MetricFunction):
    def __init__(self, span=4, max_cap=500):
        super(OverlapMetric, self).__init__()
        self.span = span
        self.max_cap = max_cap

    def calculate(self, layout: Layout, *args, **kwargs) -> float:
        greedy_sample_i, greedy_sample_j = greedy_sample(layout, self.span)
        macro_sample_i, macro_sample_j = macro_sample(layout, self.max_cap)
        overlap_x, overlap_y = calc_overlap_xy(
            layout, greedy_sample_i + macro_sample_i, greedy_sample_j + macro_sample_j)
        return float(torch.sum(overlap_x * overlap_y).cpu().clone().detach().data)

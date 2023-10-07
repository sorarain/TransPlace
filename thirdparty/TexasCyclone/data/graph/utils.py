import numpy as np
import torch
from typing import List


def pad_net_cell_list(net_cell_list: List[List[int]], truncate=-1) -> torch.Tensor:
    max_length = max([len(cell_list) for cell_list in net_cell_list])
    if truncate > 0:
        max_length = min(max_length, truncate)

    net_cell_indices_matrix = torch.zeros([len(net_cell_list), max_length], dtype=torch.int64)
    for i, cell_list in enumerate(net_cell_list):
        n_c = len(cell_list)
        if n_c > max_length:
            net_cell_indices_matrix[i, :] = torch.from_numpy(np.random.permutation(cell_list)[:max_length])
        elif n_c == max_length:
            net_cell_indices_matrix[i, :] = torch.from_numpy(np.array(cell_list))
        else:
            net_cell_indices_matrix[i, :n_c] = torch.from_numpy(np.array(cell_list))
            net_cell_indices_matrix[i, n_c:max_length] = cell_list[0]

    return net_cell_indices_matrix

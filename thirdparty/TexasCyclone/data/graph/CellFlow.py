import numpy as np
import dgl
import queue
import torch
import time
from typing import Tuple, List, Dict, Sequence, Set
from copy import deepcopy

import os, sys
sys.path.append(os.path.abspath('.'))


class CellFlow:
    def __init__(self, graph: dgl.DGLHeteroGraph, terminal_indices: Sequence[int], bfs=True):
        n_net = graph.num_nodes(ntype='net')
        n_cell = graph.num_nodes(ntype='cell')

        # 1. Get cell connection
        list_net_cells: List[Set[int]] = [set() for _ in range(n_net)]
        list_cell_nets: List[Set[int]] = [set() for _ in range(n_cell)]
        for net, cell in zip(*graph.edges(etype='pinned')):
            list_net_cells[int(net)].add(int(cell))
            list_cell_nets[int(cell)].add(int(net))

        # 2. Collect the flow edges
        flow_edge_indices: List[Tuple[int, int, int, int, int]] = []
        cell_paths: List[List[np.ndarray]] = [[] for _ in range(n_cell)]
        edge_cnt = 0

        ## 2.1 Label the terminals (fixed)
        for t in terminal_indices:
            flow_edge_indices.append((-1, -1, -1, -1, t))
            edge_cnt += 1

        ## 2.2 Find the paths from terminals to movable cells
        set_terminals = set(terminal_indices)
        if bfs:
            # Initialize father edge index for each cell
            cell_father_edge_idx = [-1] * n_cell
            for i, (_, _, c, _, ch) in enumerate(flow_edge_indices):
                cell_father_edge_idx[ch] = i
            # Construct flows
            cursor = 0
            left_net_set = set(range(n_net))
            while cursor < edge_cnt:
                _, _, father_cell, father_net, cell = flow_edge_indices[cursor]
                net_set = list_cell_nets[cell]
                valid_net_set = net_set & left_net_set
                for net in valid_net_set:
                    children = list_net_cells[net]
                    for child in children:
                        if child in set_terminals or child == cell:
                            continue
                        flow_edge_indices.append((father_cell, father_net, cell, net, child))
                        edge_cnt += 1
                        if cell_father_edge_idx[child] == -1:
                            cell_father_edge_idx[child] = edge_cnt - 1
                left_net_set -= valid_net_set
                cursor += 1

            # Collect cell paths
            for i, (fc, fn, c, n, ch) in enumerate(flow_edge_indices):
                if c == -1:
                    cell_paths[ch].append(np.array([i]))
                else:
                    cell_paths[ch].append(np.concatenate([cell_paths[c][0], np.array([i])]))
        else:
            # Initialize roots of CellFlow
            fathers_list: List[List[Tuple[int, int]]] = [[] for _ in range(n_cell)]
            for t in terminal_indices:
                fathers_list[t].append((-1, -1))

            # Expand the flow
            net_flag = [False for _ in range(n_net)]
            cell_queue = queue.Queue()
            for t in terminal_indices:
                cell_queue.put(t)
            while not cell_queue.empty():
                cell: int = cell_queue.get()
                for net in list_cell_nets[cell]:
                    if net_flag[net]:
                        continue
                    adj_cells = list_net_cells[net]
                    net_flag[net] = True
                    for adj_cell in adj_cells - {cell}:
                        if len(fathers_list[adj_cell]) == 0:
                            cell_queue.put(adj_cell)
                        fathers_list[adj_cell].append((cell, net))

            # find paths
            dict_cell_net_children: Dict[int, List[Tuple[int, int]]] = {}
            for i, fathers in enumerate(fathers_list):
                for f in fathers:
                    dict_cell_net_children.setdefault(f[0], []).append((f[1], i))
            edge_stack, temp_path = queue.LifoQueue(), []
            for i, t in enumerate(terminal_indices):
                assert edge_stack.empty()
                edge_stack.put((-1, -1, t))
                while not edge_stack.empty():
                    k = edge_stack.get()
                    if k == (-2, -2, -2):
                        temp_path.pop()
                        continue
                    if k[0] == -1:
                        temp_path.append(i)
                    else:
                        flow_edge_indices.append((fathers_list[k[0]][0][0], fathers_list[k[0]][0][1], k[0], k[1], k[2]))
                        temp_path.append(edge_cnt)
                        edge_cnt += 1
                    cell_paths[k[2]].append(np.array(deepcopy(temp_path)))
                    edge_stack.put((-2, -2, -2))
                    # Sample only one path from each of its father to avoid combination explosion
                    if k[0] == fathers_list[k[2]][0][0]:
                        for net, child in dict_cell_net_children.get(k[2], []):
                            if child in set_terminals:
                                continue
                            edge_stack.put((k[2], net, child))

        # time/space complexity: O(D * P)
        # where D is the max depth of CellFlow and P is the # of pins
        assert len(flow_edge_indices) == edge_cnt
        self.flow_edge_cnt = edge_cnt
        # cell->net->*cell*->net->cell
        self.flow_edge_indices = np.array(flow_edge_indices)
        self.cell_paths = np.array(cell_paths, dtype=object)
        """
        这里用numpy.array存储能减少内存开销
        """


if __name__ == '__main__':
    g = dgl.heterograph({
        ('net', 'pinned', 'cell'): (
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [0, 2, 3, 0, 1, 2, 4, 1, 4, 5, 4]
        ),
    }, num_nodes_dict={'cell': 6, 'net': 5})
    rs = [0, 2]
    cell_flow = CellFlow(g, rs, bfs=False)
    # print(cell_flow.fathers_list)
    for _ in range(cell_flow.flow_edge_cnt):
        print(f'{_}: {cell_flow.flow_edge_indices[_]}')
    print(cell_flow.cell_paths)
    print('#####')
    cell_flow = CellFlow(g, rs, bfs=True)
    # print(cell_flow.fathers_list)
    for _ in range(cell_flow.flow_edge_cnt):
        print(f'{_}: {cell_flow.flow_edge_indices[_]}')
    print(cell_flow.cell_paths)

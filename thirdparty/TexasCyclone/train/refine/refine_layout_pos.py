import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Set
from functools import reduce

import os, sys

sys.path.append(os.path.abspath('.'))
from data.graph import Netlist, Layout
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_ref


def select_target_box(
        density_map: np.ndarray, wh: Tuple[int, int], density_threshold: float, stride_xy: Tuple[float, float],
        stride: float, sample_num=10, momentum_per=0.2, momentum: Optional[Tuple[float, float]] = None
) -> Optional[Tuple[int, int]]:
    w, h = wh
    if w >= density_map.shape[0] or h >= density_map.shape[1]:
        return None
    density = density_map[w, h]
    if density < density_threshold:
        return None

    for _ in range(sample_num):
        flip_x, flip_y = stride_xy[0] * (2 * np.random.random() - 1), stride_xy[1] * (2 * np.random.random() - 1)
        if momentum is not None:
            flip_x += momentum[0] * stride * np.random.random() * momentum_per
            flip_y += momentum[1] * stride * np.random.random() * momentum_per
        fw, fh = w + int(flip_x), h + int(flip_y)
        if fw < 0 or fh < 0 or fw >= density_map.shape[0] or fh >= density_map.shape[1]:
            continue
        if density_map[fw, fh] < density_threshold:
            return fw, fh

    return None


def refined_layout_pos(
        layout: Layout,
        box_w=10, box_h=10, density_threshold=10.0, stride_per=0.003, momentum_per=5.0, sample_num=50, epochs=10,
        seed=0, use_momentum=True, use_tqdm=False
) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    cell_pos = np.array(layout.cell_pos.cpu().clone().detach(), dtype=np.float32)
    cell_size = np.array(layout.cell_size.cpu().clone().detach(), dtype=np.float32)
    cell_span = np.array(layout.cell_span.cpu().clone().detach(), dtype=np.float32)
    layout_size = layout.netlist.layout_size
    terminal_indices = layout.netlist.terminal_indices
    n_cell = layout.netlist.n_cell

    terminal_set = set(terminal_indices)
    movable_set = set(range(n_cell)) - terminal_set
    shape = (int(layout_size[0] / box_w) + 1, int(layout_size[1] / box_h) + 1)
    movable_density_map = np.zeros(shape=shape, dtype=np.float32)
    terminal_density_map = np.zeros(shape=shape, dtype=np.float32)
    box_size = box_w * box_h

    print(f'\t\tInitializing density map...')
    for mid in movable_set:
        if cell_pos[mid, 0] < cell_size[mid, 0] / 2:
            cell_pos[mid, 0] = cell_size[mid, 0] / 2
        if cell_pos[mid, 0] > layout_size[0] - cell_size[mid, 0] / 2:
            cell_pos[mid, 0] = layout_size[0] - cell_size[mid, 0] / 2
        if cell_pos[mid, 1] < cell_size[mid, 1] / 2:
            cell_pos[mid, 1] = cell_size[mid, 1] / 2
        if cell_pos[mid, 1] > layout_size[1] - cell_size[mid, 1] / 2:
            cell_pos[mid, 1] = layout_size[1] - cell_size[mid, 1] / 2
        w, h = int(cell_pos[mid, 0] / box_w), int(cell_pos[mid, 1] / box_h)
        assert w < shape[0] or h < shape[1]
        movable_density_map[w, h] += cell_size[mid, 0] * cell_size[mid, 1] / box_size
    print(f'\t\t\tMax movavle density: {np.max(movable_density_map):.3f}')

    for tid in terminal_set:
        span = cell_span[tid, :]
        w1, w2 = map(int, span[[0, 2]] / box_w)
        h1, h2 = map(int, span[[1, 3]] / box_h)
        w2 = min(w2 + 1, shape[0])
        h2 = min(h2 + 1, shape[1])
        if w2 <= w1 or h2 <= h1:
            continue
        terminal_density_map[w1: w2, h1: h2] = 1e5
    density_map = movable_density_map + terminal_density_map

    cell_momentum = np.zeros_like(cell_pos)
    dict_net_terminal_set: Dict[int, Set[int]] = {}
    dict_net_cell_set: Dict[int, Set[int]] = {}
    dict_movable_net_set: Dict[int, Set[int]] = {}

    if use_momentum:
        print(f'\t\tConstructing sets...')
        nets, cells = layout.netlist.graph.edges(etype='pinned')
        iter_zip = tqdm(zip(nets, cells), total=layout.netlist.n_pin) if use_tqdm else zip(nets, cells)
        for net, cell in iter_zip:
            net, cell = int(net), int(cell)
            dict_net_cell_set.setdefault(net, set()).add(cell)
            if cell in terminal_set:
                dict_net_terminal_set.setdefault(net, set()).add(cell)
            else:
                dict_movable_net_set.setdefault(cell, set()).add(net)

    def refresh_momentum(terminal_only=False):
        assert use_momentum is True
        if terminal_only:
            print(f'\t\tRefreshing momentum...')
        else:
            print(f'\t\tRefreshing momentum with movables...')
        iter_dict = tqdm(dict_movable_net_set.items(), total=len(dict_movable_net_set)) \
            if use_tqdm else dict_movable_net_set.items()
        for mid, net_set in iter_dict:
            if terminal_only:
                mts = reduce(lambda x, y: x | y, map(lambda x: dict_net_terminal_set.setdefault(x, set()), net_set))
            else:
                mts = reduce(lambda x, y: x | y, map(lambda x: dict_net_cell_set[x], net_set))
            if len(mts) == 0:
                continue
            t_pos = np.mean(cell_pos[list(mts), :], axis=0)
            delta_pos = t_pos - cell_pos[mid, :]
            cell_momentum[mid, :] = delta_pos / (np.linalg.norm(delta_pos) + 1e-3)

    for e in range(1, epochs + 1):
        stride_xy = (density_map.shape[0] * stride_per * (1 + 4 * e / epochs), 
                     density_map.shape[1] * stride_per * (1 + 4 * e / epochs))
        stride = np.sqrt(stride_xy[0] ** 2 + stride_xy[1] ** 2)
        print(f'\tEpoch {e}: stride = {stride}')
        if use_momentum:
            refresh_momentum(e <= 0.5 * epochs)
        print(f'\t\tMoving cells...')
        iter_movable_set = tqdm(movable_set, total=len(movable_set)) if use_tqdm else movable_set
        for mid in iter_movable_set:
            w, h = int(cell_pos[mid, 0] / box_w), int(cell_pos[mid, 1] / box_h)
            twh = select_target_box(
                density_map, (w, h), density_threshold, stride_xy, stride, sample_num, momentum_per,
                momentum=cell_momentum[mid, :] if use_momentum else None
            )
            if twh is None:
                continue
            tw, th = twh
            cell_pos[mid, 0], cell_pos[mid, 1] = \
                (tw + np.random.random()) * box_w, (th + np.random.random()) * box_h
            delta_density = cell_size[mid, 0] * cell_size[mid, 1] / box_size
            density_map[w, h] -= delta_density
            density_map[tw, th] += delta_density
    
    movable_density_map = density_map - terminal_density_map
    print(f'\t\t\tMax density with terminals: {np.max(density_map):.3f}')
    print(f'\t\t\tMax density without terminals: {np.max(movable_density_map):.3f}')
    np.random.set_state(state)
    return cell_pos


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    netlist_ = netlist_from_numpy_directory('../../data/test/dataset1/medium').original_netlist
    layout_ = layout_from_netlist_ref(netlist_)
    rlp = refined_layout_pos(layout_, box_w=100, box_h=100, density_threshold=0.001, stride_per=0.5, epochs=5,
                             seed=1, momentum_per=1.0, use_momentum=True, use_tqdm=True)
    print(rlp)

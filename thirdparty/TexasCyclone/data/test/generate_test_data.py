import numpy as np
import os
import json


def generate_netlist(dataset: str, name: str, pin_net_cell: np.ndarray, cell_pos: np.ndarray,
                     cell_data: np.ndarray, net_data: np.ndarray, pin_data: np.ndarray,
                     cell_clusters=None, layout_size=None):
    directory = f'{dataset}/{name}'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    np.save(f'{directory}/pin_net_cell.npy', pin_net_cell)
    np.save(f'{directory}/cell_pos.npy', cell_pos)
    np.save(f'{directory}/cell_data.npy', cell_data)
    np.save(f'{directory}/net_data.npy', net_data)
    np.save(f'{directory}/pin_data.npy', pin_data)
    if cell_clusters is not None:
        with open(f'{directory}/cell_clusters.json', 'w+') as fp:
            json.dump(cell_clusters, fp)
    if layout_size is not None:
        with open(f'{directory}/layout_size.json', 'w+') as fp:
            json.dump(layout_size, fp)


if not os.path.isdir('dataset1'):
    os.mkdir('dataset1')

with open('dataset1/dataset.json', 'w+') as fp:
    json.dump({
        'cell_dim': 3,
        'net_dim': 1,
        'pin_dim': 3,
    }, fp)

with open('dataset1/dataset.md', 'w+') as fp:
    fp.write('cell_data:\n')
    fp.write('- (0, 1): cell size (x, y)\n')
    fp.write('- 2: degree\n')
    fp.write('\n')
    fp.write('net_data:\n')
    fp.write('- 0: degree\n')
    fp.write('\n')
    fp.write('pin_data:\n')
    fp.write('- (0, 1): pin offset (x, y)\n')
    fp.write('- 2: in/out (0, 1)\n')
    fp.flush()

generate_netlist(
    dataset='dataset1', name='small',
    pin_net_cell=np.array([
        (0, 0),
        (0, 1),
        (0, 0),
        (1, 2),
    ], dtype=np.int64),
    cell_pos=np.array([
        [100, 0],
        [0, 200],
        [200, 200],
    ], dtype=np.float32),
    cell_data=np.array([
        [2, 50, 50, 1],
        [1, 10, 10, 0],
        [1, 10, 10, 0],
    ], dtype=np.float32),
    net_data=np.array([2, 2], dtype=np.float32),
    pin_data=np.array([
        [-25, 0, 0],
        [0, -5, 1],
        [25, 0, 0],
        [0, -5, 1],
    ], dtype=np.float32)
)

generate_netlist(
    dataset='dataset1', name='medium',
    pin_net_cell=np.array([
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 3),
    ], dtype=np.int64),
    cell_pos=np.array([
        [200, 0],
        [0, 200],
        [400, 200],
        [400, 0],
    ], dtype=np.float32),
    cell_data=np.array([
        [2, 50, 50, 1],
        [1, 10, 10, 0],
        [2, 10, 10, 0],
        [2, 10, 10, 0],
    ], dtype=np.float32),
    net_data=np.array([3, 2, 2], dtype=np.float32),
    pin_data=np.array([
        [0, 25, 0],
        [5, 0, 1],
        [-5, 0, 1],
        [0, -5, 0],
        [0, 5, 1],
        [50, 0, 0],
        [-5, 0, 1],
    ], dtype=np.float32),
    cell_clusters=[[2, 3]],
    layout_size=(450, 350)
)

generate_netlist(
    dataset='dataset1', name='large',
    pin_net_cell=np.array([
        (0, 0),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 4),
        (3, 1),
        (3, 4),
        (4, 5),
        (4, 4),
    ], dtype=np.int64),
    cell_pos=np.array([
        [100, 1000],
        [100, 500],
        [1000, 1000],
        [600, 600],
        [900, 500],
        [950, 100],
    ], dtype=np.float32),
    cell_data=np.array([
        [2, 60, 60, 1],
        [2, 12, 10, 0],
        [2, 50, 50, 1],
        [1, 15, 16, 0],
        [3, 10, 12, 0],
        [1, 10, 10, 0],
    ], dtype=np.float32),
    net_data=np.array([3, 2, 2, 2, 2], dtype=np.float32),
    pin_data=np.array([
        [30, 0, 0],
        [-25, 0, 0],
        [-5, 0, 1],
        [0, -30, 0],
        [0, 8, 1],
        [0, -25, 0],
        [0, 5, 1],
        [6, 1, 0],
        [-5, -3, 1],
        [0, -6, 0],
        [-1, 5, 1],
    ], dtype=np.float32),
    cell_clusters=[[4, 5]],
    layout_size=(1100, 1100)
)

generate_netlist(
    dataset='dataset1', name='large-noclu',
    pin_net_cell=np.array([
        (0, 0),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 4),
        (3, 1),
        (3, 4),
        (4, 5),
        (4, 4),
    ], dtype=np.int64),
    cell_pos=np.array([
        [100, 1000],
        [100, 500],
        [1000, 1000],
        [600, 600],
        [900, 500],
        [950, 100],
    ], dtype=np.float32),
    cell_data=np.array([
        [2, 60, 60, 1],
        [2, 12, 10, 0],
        [2, 50, 50, 1],
        [1, 15, 16, 0],
        [3, 10, 12, 0],
        [1, 10, 10, 0],
    ], dtype=np.float32),
    net_data=np.array([3, 2, 2, 2, 2], dtype=np.float32),
    pin_data=np.array([
        [30, 0, 0],
        [-25, 0, 0],
        [-5, 0, 1],
        [0, -30, 0],
        [0, 8, 1],
        [0, -25, 0],
        [0, 5, 1],
        [6, 1, 0],
        [-5, -3, 1],
        [0, -6, 0],
        [-1, 5, 1],
    ], dtype=np.float32),
    layout_size=(1100, 1100)
)

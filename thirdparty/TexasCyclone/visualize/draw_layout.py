import numpy
import torch
from matplotlib import pyplot as plt

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph import Netlist, Layout
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_ref


def draw_detailed_layout(layout: Layout, title='default', directory='visualize/layouts'):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    cell_pos = layout.cell_pos
    cell_size = layout.netlist.cell_prop_dict['size']
    cell_pos_corner = cell_pos - cell_size / 2
    fig = plt.figure()
    ax = plt.subplot(111)
    xs = (cell_pos - cell_size / 2)[:, 0].tolist() + (cell_pos + cell_size / 2)[:, 0].tolist()
    ys = (cell_pos - cell_size / 2)[:, 1].tolist() + (cell_pos + cell_size / 2)[:, 1].tolist()
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    scale_x, scale_y = max_x - min_x, max_y - min_y
    ax.set_xlim(min_x - 0.1 * scale_x, max_x + 0.1 * scale_x)
    ax.set_ylim(min_y - 0.1 * scale_y, max_y + 0.1 * scale_y)

    for i in range(layout.netlist.n_cell):
        ax.add_patch(plt.Rectangle(
            tuple(cell_pos_corner[i, :].tolist()),
            float(cell_size[i, 0]),
            float(cell_size[i, 1]),
            fill=False, color='red'
        ))

    fathers, sons = layout.netlist.graph.edges(etype='points-to')
    for f, s in zip(fathers.tolist(), sons.tolist()):
        ax.plot([cell_pos[f, 0], cell_pos[s, 0]], [cell_pos[f, 1], cell_pos[s, 1]], color='black')

    plt.savefig(f'{directory}/{title}.png')


if __name__ == '__main__':
    layout_ = layout_from_netlist_ref(netlist_from_numpy_directory('../data/test/dataset1/medium'))
    draw_detailed_layout(layout_, directory='layouts')

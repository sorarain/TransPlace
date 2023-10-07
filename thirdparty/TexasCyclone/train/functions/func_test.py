import torch

import os, sys
sys.path.append(os.path.abspath('.'))
from data.load_data import layout_from_directory
from visualize.draw_layout import draw_detailed_layout
from train.functions import HPWLLoss, AreaLoss, SampleOverlapLoss


if __name__ == '__main__':
    # layout, d_loss = layout_from_netlist_dis_angle(
    #     netlist_from_numpy_directory_old('../../data/test-old', 900),
    #     torch.tensor([400, 400, 400], dtype=torch.float32),
    #     torch.tensor([0, 0, -0.25], dtype=torch.float32),
    #     torch.tensor([20, 20, 200, 400, 200, 200, 200], dtype=torch.float32),
    #     torch.tensor([-1, 0.5, -0.5, -0.8, 1.5, 1.3, 1.7], dtype=torch.float32),
    # )

    layout = layout_from_directory('../../../Placement-datasets/ispd2015/mgc_des_perf_1')
    d_loss = 0.

    print(f'The netlist has {layout.netlist.n_cell} cells, {layout.netlist.n_net} nets, {layout.netlist.n_pin} pins')
    print(layout.netlist.graph)
    # print(layout.netlist.graph.edges(etype='father'))
    # print(layout.netlist.graph.edges(etype='son'))
    # print(layout.net_pos)
    print(layout.cell_pos)
    # draw_layout(layout, directory='../../visualize/layouts')

    print('Losses:')
    hpwl_loss_op = HPWLLoss()
    area_loss_op = AreaLoss()
    overlap_loss_op = SampleOverlapLoss(span=4)

    print(f'Discrepancy Loss: {d_loss}')
    print(f'HPWL Loss: {hpwl_loss_op(layout)}')
    print(f'Area Loss: {area_loss_op(layout, limit=[-500, -500, 500, 500])}')
    print(f'Overlap Loss: {overlap_loss_op(layout)}')

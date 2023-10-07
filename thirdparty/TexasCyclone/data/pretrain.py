import pickle
import torch
import numpy as np
from typing import Tuple, Dict

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph import Netlist, expand_netlist
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_ref

TOKEN = 'dis_deflect'
DIS_ANGLE_TYPE = Tuple[torch.Tensor, torch.Tensor]


def dump_pretrain_data(dir_name: str, save_type=1):
    netlist = netlist_from_numpy_directory(dir_name, save_type=save_type)
    dict_netlist = expand_netlist(netlist)

    dict_nid_dis_deflect = {}
    for nid, sub_netlist in dict_netlist.items():
        layout = layout_from_netlist_ref(sub_netlist)
        fathers, sons = layout.netlist.graph.edges(etype='points-to')
        fathers_, grandfathers = layout.netlist.graph.edges(etype='pointed-from')
        assert torch.equal(fathers, fathers_)
        terminal_son_edge_indices = []
        true_grandfather_indices = layout.netlist.cell_flow.flow_edge_indices[
                                   len(layout.netlist.terminal_indices):, 0]
        for i in range(len(fathers)):
            if true_grandfather_indices[i] == -1:
                terminal_son_edge_indices.append(i)
        terminal_son_edge_indices = torch.tensor(terminal_son_edge_indices, dtype=torch.int64)

        grandfathers_pos = layout.cell_pos[grandfathers, :]
        fathers_pos = layout.cell_pos[fathers, :]
        sons_pos = layout.cell_pos[sons, :]
        grandfathers_pos[terminal_son_edge_indices, 0] = layout.netlist.layout_size[0] / 2
        grandfathers_pos[terminal_son_edge_indices, 1] = layout.netlist.layout_size[1] / 2

        edge_rel_pos = sons_pos - fathers_pos
        former_edge_rel_pos = fathers_pos - grandfathers_pos
        edge_dis = torch.norm(edge_rel_pos, dim=1)
        edge_angle = torch.arctan2(edge_rel_pos[:, 1], edge_rel_pos[:, 0])
        former_edge_angle = torch.arctan2(former_edge_rel_pos[:, 1], former_edge_rel_pos[:, 0])
        former_edge_angle[terminal_son_edge_indices] += np.pi

        edge_deflect = edge_angle - former_edge_angle
        dict_nid_dis_deflect[nid] = (edge_dis, edge_deflect)

    with open(f'{dir_name}/{TOKEN}.pkl', 'wb+') as fp:
        pickle.dump(dict_nid_dis_deflect, fp)


def load_pretrain_data(dir_name: str, save_type=1) -> Tuple[Netlist, Dict[int, DIS_ANGLE_TYPE]]:
    # save_type 2: force save
    pretrain_pickle_path = f'{dir_name}/{TOKEN}.pkl'
    if not os.path.exists(pretrain_pickle_path) or save_type != 1:
        dump_pretrain_data(dir_name, save_type=save_type)
    print(f'\tLoading {pretrain_pickle_path}...')
    with open(pretrain_pickle_path, 'rb') as fp:
        dict_nid_dis_deflect = pickle.load(fp)
    netlist = netlist_from_numpy_directory(dir_name, save_type=1)
    return netlist, dict_nid_dis_deflect

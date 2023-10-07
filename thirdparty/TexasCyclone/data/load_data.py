import json
import numpy as np
import torch
import pickle
import dgl
from typing import Tuple
from copy import deepcopy

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.Netlist import Netlist
from data.graph.Layout import Layout


def netlist_from_numpy_directory(
        dir_name: str,
        use_hierarchical: bool = True,
        save_type: int = 1,
) -> Netlist:
    # 0: ignore cache; 1: use and dump cache; 2: force dump cache
    print(f'\tLoading {dir_name}')
    netlist_pickle_path = f'{dir_name}/netlist.pkl'
    if save_type == 1 and os.path.exists(netlist_pickle_path):
        with open(netlist_pickle_path, 'rb') as fp:
            netlist = pickle.load(fp)
        return netlist
    pin_net_cell = np.load(f'{dir_name}/pin_net_cell.npy')
    cell_data = np.load(f'{dir_name}/cell_data.npy')
    net_data = np.load(f'{dir_name}/net_data.npy')
    pin_data = np.load(f'{dir_name}/pin_data.npy')

    n_cell, n_net = cell_data.shape[0], net_data.shape[0]
    if os.path.exists(f'{dir_name}/cell_pos.npy'):
        cells_pos_corner = np.load(f'{dir_name}/cell_pos.npy')
    else:
        cells_pos_corner = np.zeros(shape=[n_cell, 2], dtype=np.float)
    cells = list(pin_net_cell[:, 1])
    nets = list(pin_net_cell[:, 0])
    cells_size = torch.tensor(cell_data[:, [1, 2]], dtype=torch.float32)
    cells_degree = torch.tensor(cell_data[:, 0], dtype=torch.float32).unsqueeze(-1)
    cells_type = torch.tensor(cell_data[:, 3], dtype=torch.float32).unsqueeze(-1)
    cells_ref_pos = torch.tensor(cells_pos_corner, dtype=torch.float32) + cells_size / 2
    cells_pos = cells_ref_pos.clone()
    cells_pos[cells_type[:, 0] < 1e-5, :] = torch.nan
    nets_degree = torch.tensor(net_data, dtype=torch.float32).unsqueeze(-1)
    pins_pos = torch.tensor(pin_data[:, [0, 1]], dtype=torch.float32)
    pins_io = torch.tensor(pin_data[:, 2], dtype=torch.float32).unsqueeze(-1)

    if os.path.exists(f'{dir_name}/cell_clusters.json'):
        with open(f'{dir_name}/cell_clusters.json') as fp:
            cell_clusters = json.load(fp)
    else:
        cell_clusters = None
    if os.path.exists(f'{dir_name}/layout_size.json'):
        with open(f'{dir_name}/layout_size.json') as fp:
            layout_size = json.load(fp)
    else:
        cells_pos_up_corner = cells_ref_pos + cells_size / 2
        layout_size = tuple(map(float, torch.max(cells_pos_up_corner[cells_type[:, 0] > 0.5], dim=0)[0]))

    graph: dgl.DGLHeteroGraph = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells, nets),
        ('net', 'pinned', 'cell'): (nets, cells),
        ('cell', 'points-to', 'cell'): ([], []),
        ('cell', 'pointed-from', 'cell'): ([], []),
        ('cell', 'points-to-net', 'net'): ([], []),
        ('cell', 'pointed-from-net', 'net'): ([], []),
    }, num_nodes_dict={'cell': n_cell, 'net': n_net})

    cells_feat = torch.cat([torch.log(cells_size), cells_degree], dim=-1)
    nets_feat = torch.cat([nets_degree], dim=-1)
    pins_feat = torch.cat([pins_pos / 1000, pins_io], dim=-1)

    graph.nodes['cell'].data['ref_pos'] = cells_ref_pos
    graph.nodes['cell'].data['pos'] = cells_pos
    graph.nodes['cell'].data['size'] = cells_size
    graph.nodes['cell'].data['feat'] = cells_feat
    graph.nodes['cell'].data['type'] = cells_type
    graph.nodes['net'].data['degree'] = nets_degree
    graph.nodes['net'].data['feat'] = nets_feat
    graph.edges['pinned'].data['pos'] = pins_pos
    graph.edges['pinned'].data['io'] = pins_io
    graph.edges['pinned'].data['feat'] = pins_feat

    netlist = Netlist(
        graph=graph,
        layout_size=layout_size,
        hierarchical=use_hierarchical and cell_clusters is not None and len(cell_clusters),
        cell_clusters=cell_clusters,
        original_netlist=Netlist(
            graph=deepcopy(graph),
            layout_size=layout_size, simple=True
        )
    )
    if save_type != 0:
        with open(netlist_pickle_path, 'wb+') as fp:
            pickle.dump(netlist, fp)
    return netlist


def layout_from_netlist_dis_deflect(
        netlist: Netlist,
        movable_edge_dis: torch.Tensor, movable_edge_deflect: torch.Tensor,
) -> Tuple[Layout, torch.Tensor]:
    device = movable_edge_dis.device
    edge_deflect = torch.cat([netlist.terminal_edge_theta_rev.to(device), movable_edge_deflect])
    path_angle = netlist.path_edge_matrix.to(device) @ edge_deflect
    edge_angle = path_angle[netlist.edge_ends_path_indices.to(device)]
    movable_edge_angle = edge_angle[len(netlist.terminal_indices):]
    movable_edge_rel_pos = torch.stack([movable_edge_dis * torch.cos(movable_edge_angle),
                                        movable_edge_dis * torch.sin(movable_edge_angle)]).t()
    edge_rel_pos = torch.vstack([netlist.terminal_edge_rel_pos.to(device), movable_edge_rel_pos])
    cell_rel_pos = netlist.cell_path_edge_matrix.to(device) @ edge_rel_pos
    path_rel_pos = netlist.path_edge_matrix.to(device) @ edge_rel_pos
    path_rel_pos_discrepancy = path_rel_pos - netlist.path_cell_matrix.to(device) @ cell_rel_pos
    center_pos = torch.tensor(netlist.layout_size, dtype=torch.float32).to(device) / 2
    return Layout(netlist, cell_rel_pos + center_pos), torch.mean(torch.norm(path_rel_pos_discrepancy, dim=1))


def layout_from_netlist_cell_pos(netlist: Netlist, cell_pos: torch.Tensor):
    return Layout(netlist, cell_pos)


def layout_from_netlist_ref(netlist: Netlist) -> Layout:
    return Layout(netlist, netlist.graph.nodes['cell'].data['ref_pos'])


if __name__ == '__main__':
    # netlist_ = netlist_from_numpy_directory('test/dataset1/large', save_type=2)
    netlist_ = netlist_from_numpy_directory('../../Placement-datasets/dac2012/superblue2', save_type=2)
    print(netlist_.original_netlist.graph)
    print(netlist_.graph)
    # print(netlist_.cell_prop_dict)
    # print(netlist_.net_prop_dict)
    # print(netlist_.pin_prop_dict)
    # print(netlist_.n_cell, netlist_.n_net, netlist_.n_pin)
    # print(netlist_.cell_flow.fathers_list)
    # for _, edges in enumerate(netlist_.cell_flow.flow_edge_indices):
    #     print(f'{_}: {edges}')
    print('flow edges:', len(netlist_.cell_flow.flow_edge_indices))
    print('path depth:', sum(map(lambda x: sum(map(lambda y: len(y), x)), netlist_.cell_flow.cell_paths)))
    # print(netlist_.cell_flow.cell_paths)
    # print(netlist_.cell_path_edge_matrix.to_dense().numpy())
    # print(netlist_.path_cell_matrix.to_dense().numpy())
    # print(netlist_.path_edge_matrix.to_dense().numpy())
    # print(netlist_.graph.edges(etype='points-to'))
    # print(netlist_.graph.edges['points-to'].data)
    cnt = 0
    for k, v in netlist_.dict_sub_netlist.items():
        print(f'{k}:')
        print('\t', v.graph)
        # print('\t', v.cell_prop_dict)
        # print('\t', v.net_prop_dict)
        # print('\t', v.pin_prop_dict)
        # print('\t', v.n_cell, v.n_net, v.n_pin)
        # print('\t', v.cell_flow.fathers_list)
        # for _, edges in enumerate(v.cell_flow.flow_edge_indices):
        #     print('\t', f'{_}: {edges}')
        print('\tflow edges:', len(v.cell_flow.flow_edge_indices))
        print('\tpath depth:', sum(map(lambda x: sum(map(lambda y: len(y), x)), v.cell_flow.cell_paths)))
        # print('\t', v.cell_flow.cell_paths)
        # print(v.cell_path_edge_matrix.to_dense().numpy())
        # print(v.path_cell_matrix.to_dense().numpy())
        # print(v.path_edge_matrix.to_dense().numpy())
        # print(v.graph.edges(etype='points-to'))
        # print(v.graph.edges['points-to'].data)
        cnt += 1
        if cnt >= 10:
            break

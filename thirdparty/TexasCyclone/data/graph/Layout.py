import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
from typing import Dict, List, Tuple, Optional, Any

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.Netlist import Netlist, expand_netlist


class Layout:
    def __init__(
            self, netlist: Netlist,
            cell_pos: torch.Tensor = None,
    ):
        self.netlist = netlist
        self._cell_pos = cell_pos
        self._cell_span = None  # (x1, y1, x2, y2)
        self._net_span = None  # (x1, y1, x2, y2)

    @property
    def cell_size(self) -> torch.Tensor:
        return self.netlist.graph.nodes['cell'].data['size']
        
    @property
    def cell_pos(self) -> Optional[torch.Tensor]:
        assert self._cell_pos is not None
        return self._cell_pos

    @property
    def cell_span(self) -> Optional[torch.Tensor]:
        if self._cell_span is None:
            cell_pos = self.cell_pos
            cell_size = self.cell_size.to(cell_pos.device)
            x1_y1 = cell_pos - cell_size / 2
            x2_y2 = cell_pos + cell_size / 2
            self._cell_span = torch.cat([x1_y1, x2_y2], dim=-1)
        return self._cell_span

    @property
    def net_span(self) -> Optional[torch.Tensor]:
        if self._net_span is None:
            net_cell_indices_matrix = self.netlist.net_cell_indices_matrix
            cell_span = self.cell_span
            net_cell_span = cell_span[net_cell_indices_matrix, :]
            self._net_span = torch.cat([
                torch.min(net_cell_span[:, :, :2], dim=1)[0],
                torch.max(net_cell_span[:, :, 2:], dim=1)[0]
            ], dim=-1)
        return self._net_span


def assemble_layout(dict_layout: Dict[int, Layout], device) -> Layout:
    # key is the id of pseudo macro in main netlist
    # main netlist with key -1
    dict_netlist = expand_netlist(dict_layout[-1].netlist)
    original_netlist: Netlist = dict_netlist[-1].original_netlist
    cell_pos = torch.zeros(
        size=[original_netlist.graph.num_nodes(ntype='cell') + len(dict_layout) - 1, 2],
        dtype=torch.float32, device=device
    )
    if len(dict_layout) == 1:
        layout = dict_layout[-1]
    else:
        for nid, sub_layout in dict_layout.items():
            sub_netlist = dict_netlist[nid]
            cell_pos[sub_netlist.graph.nodes['cell'].data[dgl.NID], :] = sub_layout.cell_pos
        layout = Layout(original_netlist, cell_pos[:original_netlist.graph.num_nodes(ntype='cell'), :])
    return layout


def assemble_layout_with_netlist_info(dict_netlist_info: Dict[int, Dict[str, Any]], dict_netlist: Dict[int, Netlist],
                                      device) -> Layout:
    original_netlist: Netlist = dict_netlist[-1].original_netlist
    cell_pos = torch.zeros(
        size=[original_netlist.graph.num_nodes(ntype='cell') + len(dict_netlist_info) - 1, 2],
        dtype=torch.float32, device=device
    )
    if len(dict_netlist_info) == 1:
        # TODO: 直接生成layout
        return Layout(original_netlist,dict_netlist_info[-1]['cell_pos'])
    else:
        for nid, sub_netlist in dict_netlist.items():
            if nid == -1:
                cell_pos[sub_netlist.graph.nodes['cell'].data[dgl.NID], :] = \
                    dict_netlist_info[nid]['cell_pos'].to(device)
            else:
                abs_cell_pos = dict_netlist_info[nid]['cell_pos'].to(device) + \
                               dict_netlist_info[-1]['cell_pos'][nid, :].to(device) - \
                               torch.tensor(dict_netlist[nid].layout_size, dtype=torch.float32, device=device) / 2
                cell_pos[sub_netlist.graph.nodes['cell'].data[dgl.NID], :] = abs_cell_pos
        layout = Layout(original_netlist, cell_pos[:original_netlist.graph.num_nodes(ntype='cell'), :])
    return layout


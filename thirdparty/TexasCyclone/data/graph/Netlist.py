from tkinter import E
import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
import tqdm
from time import time
from typing import Dict, List, Tuple, Optional

from pympler import asizeof
import psutil
from memory_profiler import profile, memory_usage
import copy
import ctypes

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.CellFlow import CellFlow
from data.graph.utils import pad_net_cell_list

import json
total_time = 0.0
def timer(func):
    def fn2(*args,**kwargs):
        f=time()
        result = func(*args,**kwargs)
        d = time()
        c = d-f
        global total_time
        total_time += c
        return result
    return fn2

class Netlist:
    def __init__(
            self, graph: dgl.DGLHeteroGraph,
            layout_size: Optional[Tuple[float, float]] = None,
            hierarchical: bool = False,
            cell_clusters: Optional[List[List[int]]] = None,
            original_netlist=None, simple=False, temp_device='cuda:0'
    ):
        #######################################################
        # main properties
        self.device = torch.device(temp_device)
        self.graph = graph.to(self.device)
        self.original_netlist = original_netlist
        self.dict_sub_netlist: Dict[int, Netlist] = {}

        #######################################################
        # adapt hierarchy
        if hierarchical:
            self.adapt_hierarchy(cell_clusters, use_tqdm=True)
        self.n_cell = self.graph.num_nodes(ntype='cell')
        self.n_net = self.graph.num_nodes(ntype='net')
        self.n_pin = self.graph.num_edges(etype='pinned')

        #######################################################
        # adapt layout size
        self.layout_size = layout_size
        if self.layout_size is None:
            self.adapt_layout_size()
            assert self.layout_size is not None

        #######################################################
        # adapt terminals
        self.terminal_indices = list(map(lambda x: int(x),
                                         torch.argwhere(self.graph.nodes['cell'].data['type'][:, 0] > 0).view(-1)))
        if len(self.terminal_indices) == 0:
            self.adapt_terminals()
            assert len(self.terminal_indices) > 0
            
        self._net_cell_indices_matrix = None

        #######################################################
        # simple netlists need no cell flow
        if simple:
            return
        self._cell_flow = None
        self._cell_path_edge_matrix = None
        self._path_cell_matrix = None
        self._path_edge_matrix = None
        self._edge_ends_path_indices = None
        ##############
        self._mask_edge_indices = None
        ##############
        self.terminal_edge_rel_pos = self.graph.nodes['cell'].data['pos'][self.terminal_indices, :].to(
            'cpu') - torch.tensor(self.layout_size, dtype=torch.float32) / 2
        self.terminal_edge_theta_rev = torch.arctan2(self.terminal_edge_rel_pos[:, 1],
                                                     self.terminal_edge_rel_pos[:, 0]) + torch.pi
        self.n_flow_edge = len(self.cell_flow.flow_edge_indices)
        grandfathers, gf_nets, fathers, fs_nets, sons = zip(
            *self.cell_flow.flow_edge_indices[len(self.terminal_indices):])
        grandfathers, gf_nets = list(grandfathers), list(gf_nets)
        for i in range(len(fathers)):
            if grandfathers[i] == -1:
                grandfathers[i] = fathers[i]
                gf_nets[i] = fs_nets[i]
        self.graph.add_edges(fathers, sons, etype='points-to')
        self.graph.add_edges(fathers, grandfathers, etype='pointed-from')
        self.graph.add_edges(fathers, fs_nets, etype='points-to-net')
        self.graph.add_edges(fathers, gf_nets, etype='pointed-from-net')

        assert self.cell_path_edge_matrix is not None
        if hierarchical:
            print('\t\ttotal size:', asizeof.asizeof(self) / 2 ** 20)
            print('\t\tgraph size:', asizeof.asizeof(self.graph) / 2 ** 20)
            print('\t\toriginal_netlist size:', asizeof.asizeof(self.original_netlist) / 2 ** 20)
            print('\t\tdict_sub_netlist size:', asizeof.asizeof(self.dict_sub_netlist) / 2 ** 20)
            print('\t\tcell_flow size:', asizeof.asizeof(self.cell_flow) / 2 ** 20)
            print('\t\tcell_path_edge_matrix size:', asizeof.asizeof(self.cell_path_edge_matrix) / 2 ** 20)
            print('\t\tpath_cell_matrix size:', asizeof.asizeof(self.path_cell_matrix) / 2 ** 20)
            print('\t\tpath_edge_matrix size:', asizeof.asizeof(self.path_edge_matrix) / 2 ** 20)
            print('\t\tedge_ends_path_indices size:', asizeof.asizeof(self.edge_ends_path_indices) / 2 ** 20)
            global total_time
            if os.path.exists("/root/DREAMPlace/time/cellflow_time.json"):
                with open("/root/DREAMPlace/time/cellflow_time.json","r") as f:
                    data = json.load(f)
                for k,v in data.items():
                    if v == -1:
                        data[k] = total_time
                        total_time = 0.0
                        with open("/root/DREAMPlace/time/cellflow_time.json","w") as f:
                            f.write(json.dumps(data))
                        break

    @property
    def cell_flow(self) -> CellFlow:
        if self._cell_flow is None:
            self.construct_cell_flow()
        return self._cell_flow

    @property
    def cell_path_edge_matrix(self) -> torch.sparse.Tensor:
        if self._cell_path_edge_matrix is None:
            self.construct_cell_path_edge_matrices()
        return self._cell_path_edge_matrix

    @property
    def path_cell_matrix(self) -> torch.sparse.Tensor:
        if self._path_cell_matrix is None:
            self.construct_cell_path_edge_matrices()
        return self._path_cell_matrix

    @property
    def path_edge_matrix(self) -> torch.sparse.Tensor:
        if self._path_edge_matrix is None:
            self.construct_cell_path_edge_matrices()
        return self._path_edge_matrix

    @property
    def edge_ends_path_indices(self) -> torch.Tensor:
        if self._edge_ends_path_indices is None:
            self.construct_cell_path_edge_matrices()
        return self._edge_ends_path_indices

    @property
    def net_cell_indices_matrix(self) -> Optional[torch.Tensor]:
        if self._net_cell_indices_matrix is None:
            ncl = [[] for _ in range(self.n_net)]
            nets, cells = self.graph.edges(etype='pinned')
            net_cell_tuples = list(zip(nets, cells))
            for net, cell in net_cell_tuples:
                ncl[int(net)].append(int(cell))
            self._net_cell_indices_matrix = pad_net_cell_list(ncl, 30)
        return self._net_cell_indices_matrix
    
    ##############
    @property
    def mask_edge_indices(self) -> torch.Tensor:
        if self._mask_edge_indices is None:
            self.construct_cell_path_edge_matrices()
        return self._mask_edge_indices
    ##############

    def get_cell_clusters(self) -> List[List[int]]:
        raise NotImplementedError

    def adapt_hierarchy(self, cell_clusters: Optional[List[List[int]]], use_tqdm=False):
        if cell_clusters is None:
            cell_clusters = self.get_cell_clusters()

        temp_n_cell = self.graph.num_nodes(ntype='cell')
        parted_cells = set()

        sub_graph_net_degree_dict_list = [dict() for _ in range(len(cell_clusters))]
        belong_node = np.ones(temp_n_cell) * -1
        for i, sub_graph_list in enumerate(cell_clusters):
            for node in sub_graph_list:
                belong_node[int(node)] = i
        for net_id, cell_id in zip(*[ns.tolist() for ns in self.graph.edges(etype='pinned')]):
            sub_graph_id = int(belong_node[int(cell_id)])
            if sub_graph_id == -1:
                continue
            sub_graph_net_degree_dict_list[sub_graph_id].setdefault(int(net_id), 0)
            sub_graph_net_degree_dict_list[sub_graph_id][int(net_id)] += 1

        all_pseudo_cell_cnt = 0
        all_pseudo_cell_ref_pos = torch.zeros(size=[0, 2], dtype=torch.float32, device=self.device)
        all_pseudo_cell_pos = torch.zeros(size=[0, 2], dtype=torch.float32, device=self.device)
        all_pseudo_cell_size = torch.zeros(size=[0, 2], dtype=torch.float32, device=self.device)
        all_pseudo_cell_feat = None
        all_pseudo_pin_cell = torch.zeros(size=[0], dtype=torch.int64, device=self.device)
        all_pseudo_pin_net = torch.zeros(size=[0], dtype=torch.int64, device=self.device)
        all_pseudo_pin_pos = torch.zeros(size=[0, 2], dtype=torch.float32, device=self.device)
        all_pseudo_pin_io = torch.zeros(size=[0, 1], dtype=torch.float32, device=self.device)
        all_pseudo_pin_feat = None
        iter_partition_list = tqdm.tqdm(cell_clusters, total=len(cell_clusters)) if use_tqdm else cell_clusters
        for i, partition in enumerate(iter_partition_list):
            if len(partition) <= 1:
                continue
            partition_set = set(partition)
            parted_cells |= partition_set

            new_net_degree_dict = sub_graph_net_degree_dict_list[i]
            keep_nets_id = torch.tensor(list(new_net_degree_dict.keys()), dtype=torch.int64, device=self.device)
            keep_nets_degree = torch.tensor(list(new_net_degree_dict.values()), dtype=torch.int64, device=self.device)
            good_nets = torch.abs(self.graph.nodes['net'].data['degree'][keep_nets_id, 0] - keep_nets_degree) < 1e-5
            good_nets_id = keep_nets_id[good_nets]  # numpy似乎不支持用TRUE FALSE来筛选数据所以换成tensor
            sub_graph = dgl.node_subgraph(self.graph, nodes={'cell': partition, 'net': keep_nets_id},
                                          output_device=self.device)
            sub_netlist = Netlist(graph=sub_graph)

            ref_pos = torch.mean(sub_netlist.graph.nodes['cell'].data['ref_pos'], dim=0)
            sub_netlist.graph.nodes['cell'].data['ref_pos'] -= \
                ref_pos - torch.tensor(sub_netlist.layout_size, dtype=torch.float32, device=self.device) / 2
            pseudo_cell_ref_pos = ref_pos.unsqueeze(dim=0)
            pseudo_cell_pos = torch.full_like(pseudo_cell_ref_pos, fill_value=torch.nan)
            pseudo_cell_size = torch.tensor(sub_netlist.layout_size,
                                            dtype=torch.float32, device=self.device).unsqueeze(dim=0)
            pseudo_cell_degree = torch.tensor([[len(keep_nets_id) - len(good_nets_id)]],
                                              dtype=torch.float32, device=self.device)
            pseudo_cell_feat = torch.cat([torch.log(pseudo_cell_size), pseudo_cell_degree], dim=-1)
            pseudo_pin_pos = torch.zeros([len(keep_nets_id), 2], dtype=torch.float32, device=self.device)
            pseudo_pin_io = torch.full(size=[len(keep_nets_id), 1], fill_value=2,
                                       dtype=torch.float32, device=self.device)
            pseudo_pin_feat = torch.cat([pseudo_pin_pos / 1000, pseudo_pin_io], dim=-1)

            all_pseudo_cell_cnt += 1
            all_pseudo_cell_ref_pos = torch.vstack([all_pseudo_cell_ref_pos, pseudo_cell_ref_pos])
            all_pseudo_cell_pos = torch.vstack([all_pseudo_cell_pos, pseudo_cell_pos])
            all_pseudo_cell_size = torch.vstack([all_pseudo_cell_size, pseudo_cell_size])
            if all_pseudo_cell_feat is None:
                all_pseudo_cell_feat = pseudo_cell_feat
            else:
                all_pseudo_cell_feat = torch.vstack([all_pseudo_cell_feat, pseudo_cell_feat])
            all_pseudo_pin_cell = torch.cat([
                all_pseudo_pin_cell,
                torch.full(size=[len(keep_nets_id)], fill_value=temp_n_cell, dtype=torch.int64, device=self.device)
            ])
            all_pseudo_pin_net = torch.cat([all_pseudo_pin_net, keep_nets_id])
            all_pseudo_pin_pos = torch.vstack([all_pseudo_pin_pos, pseudo_pin_pos])
            all_pseudo_pin_io = torch.vstack([all_pseudo_pin_io, pseudo_pin_io])
            if all_pseudo_pin_feat is None:
                all_pseudo_pin_feat = pseudo_pin_feat
            else:
                all_pseudo_pin_feat = torch.vstack([all_pseudo_pin_feat, pseudo_pin_feat])

            self.dict_sub_netlist[temp_n_cell] = sub_netlist
            temp_n_cell += 1
            sub_netlist.graph = sub_netlist.graph.cpu()

        self.graph.add_nodes(all_pseudo_cell_cnt, ntype='cell', data={
            'ref_pos': all_pseudo_cell_ref_pos,
            'pos': all_pseudo_cell_pos,
            'size': all_pseudo_cell_size,
            'feat': all_pseudo_cell_feat,
            'type': torch.zeros(size=[all_pseudo_cell_cnt, 1], dtype=torch.float32, device=self.device),
        })
        self.graph.add_edges(all_pseudo_pin_net, all_pseudo_pin_cell, etype='pinned', data={
            'pos': all_pseudo_pin_pos,
            'io': all_pseudo_pin_io,
            'feat': all_pseudo_pin_feat,
        })
        self.graph.add_edges(all_pseudo_pin_cell, all_pseudo_pin_net, etype='pins')
        left_cells = set(range(temp_n_cell)) - parted_cells
        left_nets = set()
        for net_id, cell_id in zip(*[ns.tolist() for ns in self.graph.edges(etype='pinned')]):
            if cell_id in left_cells:
                left_nets.add(net_id)
        self.graph = dgl.node_subgraph(self.graph, nodes={'cell': list(left_cells), 'net': list(left_nets)})
        self.graph = self.graph.to('cpu')
        dict_reverse_nid = {int(idx): i for i, idx in enumerate(self.graph.nodes['cell'].data[dgl.NID])}
        self.dict_sub_netlist = {dict_reverse_nid[k]: v for k, v in self.dict_sub_netlist.items()}

    def adapt_layout_size(self):
        # TODO: better layout adaption
        cells_size = self.graph.nodes['cell'].data['size']
        span = (cells_size[:, 0] * cells_size[:, 1]).sum() * 5
        self.layout_size = (span ** 0.5, span ** 0.5)

    def adapt_terminals(self):
        # TODO: better terminal selection
        biggest_cell_id = int(torch.argmax(torch.sum(self.graph.nodes['cell'].data['size'], dim=-1)))
        self.terminal_indices = [biggest_cell_id]
        self.graph.nodes['cell'].data['pos'][biggest_cell_id, :] = \
            self.graph.nodes['cell'].data['size'][biggest_cell_id, :] / 2

    @timer
    def construct_cell_flow(self):
        self._cell_flow = CellFlow(self.graph, self.terminal_indices)

    def construct_cell_path_edge_matrices(self):
        cell_path_edge_indices = [[], []]
        path_cell_indices = [[], []]
        path_edge_indices = [[], []]
        edge_ends_path_indices_inv = []
        cell_path_edge_values = []
        ##############
        mask_edge_indices = set()
        ##############
        n_paths = 0
        for c, paths in enumerate(self.cell_flow.cell_paths):
            n_path = len(paths)
            dict_edge_weight = {}
            for ip, path in enumerate(paths):
                edge_ends_path_indices_inv.append(path[-1])
                path_cell_indices[0].append(n_paths + ip)
                path_cell_indices[1].append(c)
                for i,e in enumerate(path):
                    dict_edge_weight.setdefault(e, 0)
                    dict_edge_weight[e] += 1 / n_path
                    path_edge_indices[0].append(n_paths + ip)
                    path_edge_indices[1].append(e)
                    ##############
                    if i == 0:
                        mask_edge_indices.add(e)
                    ##############
            for k, v in dict_edge_weight.items():
                cell_path_edge_indices[0].append(c)
                cell_path_edge_indices[1].append(k)
                cell_path_edge_values.append(v)
            n_paths += n_path
        edge_ends_path_indices = [-1] * n_paths
        for ip, e in enumerate(edge_ends_path_indices_inv):
            edge_ends_path_indices[e] = ip
        assert n_paths == self.n_flow_edge
        assert -1 not in edge_ends_path_indices
        self._cell_path_edge_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(cell_path_edge_indices, dtype=torch.int64),
            values=cell_path_edge_values,
            size=[self.n_cell, self.n_flow_edge], dtype=torch.float32
        )
        self._path_cell_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(path_cell_indices, dtype=torch.int64),
            values=torch.ones(size=[len(path_cell_indices[0])]),
            size=[n_paths, self.n_cell], dtype=torch.float32
        )
        self._path_edge_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(path_edge_indices, dtype=torch.int64),
            values=torch.ones(size=[len(path_edge_indices[0])]),
            size=[n_paths, self.n_flow_edge], dtype=torch.float32
        )
        self._edge_ends_path_indices = torch.tensor(edge_ends_path_indices, dtype=torch.int64)
        ##############
        self._mask_edge_indices = torch.tensor(list(mask_edge_indices), dtype=torch.int64)
        assert self._mask_edge_indices.max() < self.n_flow_edge,f"{self._mask_edge_indices} {self.n_flow_edge}"
        ##############


def expand_netlist(netlist: Netlist) -> Dict[int, Netlist]:
    # key is the id of pseudo macro in main netlist
    # main netlist with key -1
    dict_netlist = {-1: netlist}
    dict_netlist.update(netlist.dict_sub_netlist)
    return dict_netlist


def sequentialize_netlist(netlist: Netlist) -> List[Netlist]:
    return [netlist] + list(netlist.dict_sub_netlist.values())

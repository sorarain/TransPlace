import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from dgl.nn.pytorch import HeteroGraphConv, CFConv, GraphConv, GATConv, SAGEConv
import dgl
import os, sys
import numpy as np

sys.path.append(os.path.abspath('.'))
# from data.Netlist import netlist_from_numpy_directory

from data.graph import Netlist


class PlaceGNN(nn.Module):
    def __init__(
            self,
            raw_cell_feats: int,
            raw_net_feats: int,
            raw_pin_feats: int,
            config: Dict[str, Any]):
        super(PlaceGNN, self).__init__()
        self.device = config['DEVICE']

        self.raw_cell_feats = raw_cell_feats
        self.raw_net_feats = raw_net_feats
        self.raw_pin_feats = raw_pin_feats

        self.hidden_cell_feats = config['CELL_FEATS']
        self.hidden_net_feats = config['NET_FEATS']
        self.hidden_pin_feats = config['PIN_FEATS']
        self.num_layers = config['NUM_LAYERS']
        self.num_heads = config['NUM_HEADS']

        self.cell_lin = nn.Linear(self.raw_cell_feats, self.hidden_cell_feats)
        self.net_lin = nn.Linear(self.raw_net_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.raw_pin_feats, self.hidden_pin_feats)

        self.hetero_conv = HeteroGraphConvLayers(
            hidden_cell_feats=self.hidden_cell_feats,
            hidden_net_feats=self.hidden_net_feats,
            hidden_pin_feats=self.hidden_pin_feats,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            device=self.device)

        self.edge_dis_readout = nn.Linear(2 * self.hidden_cell_feats, 1)
        self.edge_angle_readout = nn.Linear(2 * self.hidden_cell_feats, 1)
        self.to(self.device)

    def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            feature: Tuple[torch.tensor, torch.tensor, torch.tensor],
            cell_size : torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_feat, net_feat, pin_feat = feature
        cell_feat = cell_feat.to(self.device)
        net_feat = net_feat.to(self.device)
        pin_feat = pin_feat.to(self.device)
        graph = graph.to(self.device)
        hidden_cell_feat = torch.tanh(self.cell_lin(cell_feat))
        hidden_net_feat = torch.tanh(self.net_lin(net_feat))
        hidden_pin_feat = torch.tanh(self.pin_lin(pin_feat))

        h = self.hetero_conv(
            graph,
            hidden_cell_feat,
            hidden_net_feat,
            hidden_pin_feat)
        hidden_cell_feat, hidden_net_feat = h['cell'], h['net']

        fathers, sons = graph.edges(etype='points-to')
        hidden_cell_pair_feat = torch.cat([
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :]
        ], dim=-1)
        edge_dis_ = torch.exp(-2+15 * torch.tanh(self.edge_dis_readout(hidden_cell_pair_feat))).view(-1)
        edge_angle = torch.tanh(self.edge_angle_readout(hidden_cell_pair_feat)).view(-1) * 4
        cell_size = cell_size.to(self.device)
        bound_size = (cell_size[fathers] + cell_size[sons]).to(self.device) / 2
        eps = torch.ones_like(edge_angle).to(self.device) * 1e-4
        tmp = torch.min(torch.abs(bound_size[:,0] / (torch.cos(edge_angle*np.pi)+eps)),torch.abs(bound_size[:,1] / (torch.sin(edge_angle*np.pi)+eps)))
        edge_dis = edge_dis_ + tmp
        return edge_dis, edge_angle

    def forward_with_netlist(
            self,
            netlist: Netlist,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = netlist.graph
        cell_feat = netlist.cell_prop_dict['feat']
        net_feat = netlist.net_prop_dict['feat']
        pin_feat = netlist.pin_prop_dict['feat']
        return self.forward(graph, (cell_feat, net_feat, pin_feat))


class HeteroGraphConvLayers(nn.Module):
    def __init__(
            self,
            hidden_cell_feats: int,
            hidden_net_feats: int,
            hidden_pin_feats: int,
            num_heads: int,
            num_layers: int,
            device: torch.device) -> None:

        super(HeteroGraphConvLayers, self).__init__()

        self.num_layers = num_layers
        self.hidden_cell_feats = hidden_cell_feats
        self.hidden_net_feats = hidden_net_feats
        self.hidden_pin_feats = hidden_pin_feats

        self.hetero_graph_conv_layers = nn.ModuleList([
            HeteroGraphConv({
                'pins':
                    HyperEdgeConv(node_in_feats=self.hidden_net_feats, edge_in_feats=self.hidden_pin_feats,
                                  hidden_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats,
                                  num_heads=num_heads),
                'pinned':  SAGEConv(in_feats=(self.hidden_net_feats,self.hidden_cell_feats),aggregator_type='mean',out_feats=self.hidden_cell_feats),
                # GraphConv(in_feats=self.hidden_net_feats, out_feats=self.hidden_cell_feats),
                    # CFConv(node_in_feats=self.hidden_net_feats, edge_in_feats=self.hidden_pin_feats,
                    #        hidden_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
            }, aggregate='max') for _ in range(num_layers)]).to(device)
        self.edge_weight_lin = nn.Linear(self.hidden_pin_feats,1)

        self.device = device
        self.to(self.device)

    def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            hidden_cell_feat: torch.Tensor,
            hidden_net_feat: torch.Tensor,
            hidden_pin_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        edge_weight = torch.tanh(self.edge_weight_lin(hidden_pin_feat))

        for hetero_conv_layer in self.hetero_graph_conv_layers:
            h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
            h = hetero_conv_layer.forward(graph.edge_type_subgraph(['pins']), h,
                                          mod_kwargs={'pins': {'edge_feats': hidden_pin_feat}})
            hidden_net_feat = h['net']
            h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
            h = hetero_conv_layer.forward(graph.edge_type_subgraph(['pinned']), h,
                                          mod_kwargs={'pinned': {'edge_weight': edge_weight}})
            # hidden_cell_feat, hidden_net_feat = h['cell'], h['net']
            hidden_cell_feat = h['cell']

        return {'cell': hidden_cell_feat, 'net': hidden_net_feat}


def edge_update_func(edges):
    edges.data['x'] = torch.cat([edges.src['x'], edges.data['e']], dim=1)
    return {'e_data': edges.data['x']}


def message_func(edges):
    return {'q': edges.data['q'], 'k': edges.data['k'], 'v': edges.data['v']}


class HyperEdgeConv(nn.Module):
    def __init__(self,
                 node_in_feats: int,
                 edge_in_feats: int,
                 hidden_feats: int,
                 out_feats: int,
                 num_heads: int = 4):
        super(HyperEdgeConv, self).__init__()
        self.node_in_feats = node_in_feats
        self.edge_in_feats = edge_in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.head_dim = self.hidden_feats // self.num_heads
        assert (
                self.head_dim * num_heads == self.hidden_feats
        ), "hidden_feats must be divisible by num_heads"

        self.Q_lin = nn.Linear(in_features=node_in_feats + edge_in_feats, out_features=hidden_feats)
        self.K_lin = nn.Linear(in_features=node_in_feats + edge_in_feats, out_features=hidden_feats)
        self.V_lin = nn.Linear(in_features=node_in_feats + edge_in_feats, out_features=hidden_feats)

        self.out_proj = nn.Linear(in_features=hidden_feats, out_features=out_feats)

        self.dropout_module = nn.Dropout()

    def forward(self,
                g: dgl.graph,
                node_feats: torch.Tensor,
                edge_feats: torch.Tensor):
        cell_feats, net_feats = node_feats
        with g.local_scope():
            g.nodes['cell'].data['x'] = cell_feats
            g.nodes['net'].data['x'] = net_feats
            g.edges['pins'].data['e'] = edge_feats

            g.apply_edges(edge_update_func)

            g.edata['q'] = self.Q_lin(g.edata['e_data'])
            g.edata['k'] = self.K_lin(g.edata['e_data'])
            g.edata['v'] = self.V_lin(g.edata['e_data'])

            # g.edata['q'] = F.linear(g.edata['e_data'],self.Q_lin)
            # g.edata['k'] = F.linear(g.edata['e_data'],self.K_lin)
            # g.edata['v'] = F.linear(g.edata['e_data'],self.V_lin)

            def reduce_func(nodes):
                q = nodes.mailbox['q']
                k = nodes.mailbox['k']
                v = nodes.mailbox['v']
                q = (q.contiguous().unflatten(2, (self.num_heads, self.head_dim))).permute(0, 2, 1, 3)
                k = (k.contiguous().unflatten(2, (self.num_heads, self.head_dim))).permute(0, 2, 3, 1)
                v = (v.contiguous().unflatten(2, (self.num_heads, self.head_dim))).permute(0, 2, 1, 3)
                q *= (self.head_dim ** -0.5)

                attn_weight = torch.matmul(q, k)
                attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)

                attn_weight = self.dropout_module(attn_weight)

                attn = torch.matmul(attn_weight, v)
                attn = attn.contiguous().permute(0, 2, 1, 3).flatten(2)

                out_attn = torch.mean(attn, dim=1)

                return {'attn': out_attn}

            def apply_node_func(nodes):
                return {'attn': nodes.data["attn"]}

            # g.update_all(message_func,reduce_func,apply_node_func)
            g.multi_update_all({'pins': (message_func, reduce_func)}, 'sum')
            return g.dstdata['attn']


# from train.argument import parse_train_args
#
# last_memory = 0
#
#
# def get_memory_total():
#     global last_memory
#     last_memory = 0
#     last_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
#     return last_memory
#
#
# def get_memory_diff():
#     last = last_memory
#     total = get_memory_total()
#     return total - last, total
#
#
# from torch.cuda.amp import autocast as autocast
#
# if __name__ == '__main__':
#     netlist = netlist_from_numpy_directory('../Placement-datasets/dac2012/superblue2')
#     raw_cell_feats = netlist.cell_prop_dict['feat'].shape[1]
#     raw_net_feats = netlist.net_prop_dict['feat'].shape[1]
#     raw_pin_feats = netlist.pin_prop_dict['feat'].shape[1]
#     args = parse_train_args()
#     args.device = "cuda:0"
#     device = torch.device(args.device)
#     config = {
#         'DEVICE': device,
#         'CELL_FEATS': 16,
#         'NET_FEATS': 16,
#         'PIN_FEATS': 16,
#         'NUM_LAYERS': 3,
#         'NUM_HEADS': 8,
#     }
#
#     model = PlaceGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
#     print("Model:", get_memory_diff())
#     with autocast():
#         net_dis, net_angle, pin_dis, pin_angle = model.forward(netlist)
#     print("Output:", get_memory_diff())

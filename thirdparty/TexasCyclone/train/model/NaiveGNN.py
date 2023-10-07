import torch
import torch.nn as nn
import dgl
from typing import Tuple, Dict, Any
from dgl.nn.pytorch import HeteroGraphConv, CFConv, GraphConv, SAGEConv
import numpy as np

from data.graph import Netlist
import dgl.function as fn


class NaiveGNN(nn.Module):
    def __init__(
            self,
            raw_cell_feats: int,
            raw_net_feats: int,
            raw_pin_feats: int,
            config: Dict[str, Any]
    ):
        super(NaiveGNN, self).__init__()
        self.device = config['DEVICE']
        self.raw_cell_feats = raw_cell_feats
        self.raw_net_feats = raw_net_feats
        self.raw_pin_feats = raw_pin_feats
        self.hidden_cell_feats = config['CELL_FEATS']
        self.hidden_net_feats = config['NET_FEATS']
        self.hidden_pin_feats = config['PIN_FEATS']
        self.pass_type = config['PASS_TYPE']
        if 'NUM_LAYERS' not in config:
            self.num_layers = 3
        else:
            self.num_layers = config['NUM_LAYERS']

        self.cell_lin = nn.Linear(self.raw_cell_feats + 2 * self.raw_net_feats, self.hidden_cell_feats)
        self.net_lin = nn.Linear(self.raw_net_feats + 2 * self.raw_cell_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.raw_pin_feats, self.hidden_pin_feats)

        self.position_embedding = PositionalEncoding(d_model = self.hidden_cell_feats,max_len = 2000000)
        # 这个naive模型只卷一层，所以直接这么写了。如果需要卷多层的话，建议卷积层单独写一个class，看起来更美观。
        if self.pass_type == 'bidirection':
            self.hetero_conv_list = nn.ModuleList([
                HeteroGraphConv({
                'pins': SAGEConv(in_feats=(self.hidden_cell_feats, self.hidden_net_feats), aggregator_type='mean',
                                   out_feats=self.hidden_cell_feats),#for cell flow
                # 'pins': GraphConv(in_feats=self.hidden_net_feats, out_feats=self.hidden_cell_feats),#model sensity
                # 'pins': CFConv(node_in_feats=self.hidden_cell_feats, edge_in_feats=self.hidden_pin_feats,
                #                  hidden_feats=self.hidden_net_feats, out_feats=self.hidden_net_feats),#model sensity
                'pinned': CFConv(node_in_feats=self.hidden_net_feats, edge_in_feats=self.hidden_pin_feats,
                                 hidden_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                # 'pinned': SAGEConv(in_feats=(self.hidden_cell_feats, self.hidden_net_feats), aggregator_type='mean',
                #                    out_feats=self.hidden_cell_feats),#model sensity
                'points-to': SAGEConv(in_feats=(self.hidden_cell_feats, self.hidden_cell_feats), aggregator_type='mean',
                                   out_feats=self.hidden_cell_feats),#for cell flow
                # 'points-to': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),#model sensity
            }, aggregate='mean')
            for _ in range(self.num_layers)
            ])
            self.edge_weight_lin = nn.Linear(self.hidden_pin_feats, 1)#for cell flow
        elif self.pass_type == 'single':
            self.hetero_conv = HeteroGraphConv({
                'pinned': SAGEConv(in_feats=(self.hidden_net_feats, self.hidden_cell_feats), aggregator_type='mean',
                                   out_feats=self.hidden_cell_feats),
                # 'points-to': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                # 'pointed-from': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
            }, aggregate='max')
            self.edge_weight_lin = nn.Linear(self.hidden_pin_feats, 1)
        else:
            raise NotImplementedError

        output_dim = 2 * self.hidden_cell_feats + self.hidden_net_feats
        # self.edge_dis_readout = nn.Linear(2 * self.hidden_cell_feats + self.hidden_net_feats, 1)
        self.edge_dis_readout = nn.Sequential(
            nn.Linear(output_dim,output_dim//2),
            nn.ReLU(),
            nn.Linear(output_dim//2,output_dim//4),
            nn.ReLU(),
            nn.Linear(output_dim//4,1)
        )
        output_dim = 3 * self.hidden_cell_feats + 2 * self.hidden_net_feats
        # self.edge_deflect_readout = nn.Linear(3 * self.hidden_cell_feats + 2 * self.hidden_net_feats, 1)
        self.edge_deflect_readout = nn.Sequential(
            nn.Linear(output_dim,output_dim//2),
            nn.ReLU(),
            nn.Linear(output_dim//2,output_dim//4),
            nn.ReLU(),
            nn.Linear(output_dim//4,1)
        )
        self.to(self.device)

    def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            cell_size: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = graph.to(self.device)
        graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'mean'), etype='pins')
        graph.update_all(fn.copy_u('feat', 'm'), fn.max('m', 'max'), etype='pins')
        graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'mean'), etype='pinned')
        graph.update_all(fn.copy_u('feat', 'm'), fn.max('m', 'max'), etype='pinned')
        cell_feat = torch.cat([
            graph.nodes['cell'].data['feat'],
            graph.nodes['cell'].data['mean'],
            graph.nodes['cell'].data['max'],
        ], dim=-1)
        net_feat = torch.cat([
            graph.nodes['net'].data['feat'],
            graph.nodes['net'].data['mean'],
            graph.nodes['net'].data['max'],
        ], dim=-1)
        pin_feat = graph.edges['pinned'].data['feat']

        cell_feat = cell_feat.to(self.device)
        net_feat = net_feat.to(self.device)
        pin_feat = pin_feat.to(self.device)
        hidden_cell_feat = torch.tanh(self.cell_lin(cell_feat))
        hidden_net_feat = torch.tanh(self.net_lin(net_feat))
        hidden_pin_feat = torch.tanh(self.pin_lin(pin_feat))

        h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
        graph = graph.to(self.device)
        if self.pass_type == 'bidirection':
            edge_weight = torch.tanh(self.edge_weight_lin(hidden_pin_feat))###for cell flow
            for hetero_conv in self.hetero_conv_list:
                h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
                h = hetero_conv.forward(graph.edge_type_subgraph(['pins']), h,
                                            mod_kwargs={'pins': {'edge_weight': edge_weight}})###for cell flow
                # h = hetero_conv.forward(graph.edge_type_subgraph(['pins']), h,mod_kwargs={'pins': {'edge_feats': hidden_pin_feat}})# ,mod_kwargs={'pins': {'edge_weight': edge_weight}}model sensity
                # h = hetero_conv.forward(graph.edge_type_subgraph(['pins']), h)#model sensity
                h = {'cell': hidden_cell_feat, 'net': h['net']}
                h = hetero_conv.forward(graph.edge_type_subgraph(['pinned','points-to']), h,
                                            mod_kwargs={'pinned': {'edge_feats': hidden_pin_feat}})###for cell flow
                # h = hetero_conv.forward(graph.edge_type_subgraph(['pinned','points-to']), h,
                #                             mod_kwargs={'pinned': {'edge_weight': edge_weight}})#model sensity
                # h = hetero_conv.forward(graph.edge_type_subgraph(['pinned','points-to']), h)#model sensity
                hidden_cell_feat = h['cell']
        elif self.pass_type == 'single':
            edge_weight = torch.tanh(self.edge_weight_lin(hidden_pin_feat))
            h = self.hetero_conv.forward(graph.edge_type_subgraph(['pinned']), h,
                                         mod_kwargs={'pinned': {'edge_weight': edge_weight}})
            hidden_cell_feat = h['cell']

        fathers, sons = graph.edges(etype='points-to')
        fathers1, grandfathers = graph.edges(etype='pointed-from')
        fathers2, fs_nets = graph.edges(etype='points-to-net')
        fathers3, gf_nets = graph.edges(etype='pointed-from-net')
        assert torch.equal(fathers, fathers1)
        assert torch.equal(fathers, fathers2)
        assert torch.equal(fathers, fathers3)
        hidden_cell_pair_feat = torch.cat([
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :],
            hidden_net_feat[fs_nets, :]
        ], dim=-1)
        hidden_cell_pair_feat_extend = torch.cat([
            hidden_cell_feat[grandfathers, :],
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :],
            hidden_net_feat[gf_nets, :],
            hidden_net_feat[fs_nets, :]
        ], dim=-1)
        # print(torch.max(self.edge_dis_readout(hidden_cell_pair_feat)),torch.min(self.edge_dis_readout(hidden_cell_pair_feat)))
        edge_dis_ = torch.exp(-2 + 15 * torch.tanh(self.edge_dis_readout(hidden_cell_pair_feat))).view(-1)
        edge_deflect = torch.tanh(self.edge_deflect_readout(hidden_cell_pair_feat_extend)).view(-1) * 2 * torch.pi
        cell_size = cell_size.to(self.device)
        bound_size = (cell_size[fathers] + cell_size[sons]).to(self.device) / 2
        edge_dis = edge_dis_ + torch.min(bound_size, dim=1)[0]
        return edge_dis, edge_deflect

    def forward_with_netlist(
            self,
            netlist: Netlist
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = netlist.graph
        cell_feat = netlist.cell_prop_dict['feat']
        net_feat = netlist.net_prop_dict['feat']
        pin_feat = netlist.pin_prop_dict['feat']
        return self.forward(graph, (cell_feat, net_feat, pin_feat), None)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
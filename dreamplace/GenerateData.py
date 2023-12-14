import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append('./build')
sys.path.append('./thirdparty/TexasCyclone')
import re
import math
import time
import numpy as np
import torch
import logging
import Params
import dreamplace.ops.place_io.place_io as place_io
import dreamplace.ops.fence_region.fence_region as fence_region
import pdb
import PlaceDB
import pickle
import json
from thirdparty.TexasCyclone.data.graph.Netlist import Netlist
from thirdparty.TexasCyclone.train.argument import parse_train_args
from thirdparty.TexasCyclone.data.graph import Netlist,Layout,expand_netlist
from copy import deepcopy
import dgl
from tqdm import tqdm
from grouping import create_group
import os.path as osp
from dreamplace import NonLinearPlace
import json
import Timer

def generate_data(netlist_dir:str,params,save_type=1,for_test=False):
    data_file_list = ['cell_pos.npy', 'cell_data.npy','net_data.npy','pin_data.npy','pin_net_cell.npy','cell_clusters.json','layout_size.json']
    if for_test:
        data_file_list = ['cell_clusters.json','layout_size.json','cell_pos.npy']
    flag = False
    for data_file_name in data_file_list:
        if not os.path.exists(osp.join(netlist_dir,data_file_name)) or save_type == 2:
            flag = True
    if flag:
        placedb = PlaceDB.PlaceDB()
        placedb(params)
    for data_file_name in data_file_list:
        if not os.path.exists(osp.join(netlist_dir,data_file_name)) or save_type == 2:
            if data_file_name == 'cell_data.npy':
                cells_size = torch.tensor(np.concatenate([placedb.node_x.reshape((-1,1)),placedb.node_y.reshape((-1,1))],axis=1))
                cells_degree = torch.zeros([placedb.num_physical_nodes,1])
                cells_type = torch.zeros([placedb.num_physical_nodes,1])
                fix_node_index_list = list(placedb.rawdb.fixedNodeIndices())
                for i in range(placedb.num_physical_nodes):
                    cells_size[i] = torch.tensor([placedb.node_size_x[i],placedb.node_size_y[i]]) + 1e-5
                    if i in range(placedb.num_movable_nodes):
                        cells_type[i] = 0
                    elif i in fix_node_index_list:
                        cells_type[i] = 1
                    else:
                        cells_type[i] = 2

                for i,net_pin_list in enumerate(placedb.net2pin_map):
                    for pin_id in net_pin_list:
                        parent_node_id = placedb.pin2node_map[pin_id]
                        cells_degree[parent_node_id] += 1
                np.save(osp.join(netlist_dir,data_file_name),torch.stack([cells_degree.squeeze(),cells_size[:,0],cells_size[:,1],cells_type.squeeze()],dim=1).numpy())
            elif data_file_name == 'cell_pos.npy':
                os.system(f"cp -r {netlist_dir} ./build/benchmarks")
                netlist_name = netlist_dir.split("/")[-1]
                os.system(f"mkdir -p ./result/DREAMPlaceGP/{netlist_name}")
                params.__dict__["save_gp_dir"] = f"./result/DREAMPlaceGP/{netlist_name}/{netlist_name}"
                timer = None
                if params.timing_opt_flag:
                    timer = Timer.Timer()
                    timer(params, placedb)
                    timer.update_timing()
                placer = NonLinearPlace.NonLinearPlace(params, placedb,timer)
                metrics = placer(params, placedb)
                node_x,node_y = placedb.node_x,placedb.node_y
                np.save(osp.join(netlist_dir,data_file_name),np.stack([node_x,node_y],axis=1))

            elif data_file_name == 'net_data.npy':
                nets_degree = torch.zeros([len(placedb.net_names),1])
                for i,net_pin_list in enumerate(placedb.net2pin_map):
                    nets_degree[i] = len(net_pin_list)
                np.save(osp.join(netlist_dir,data_file_name),nets_degree.squeeze().numpy())
            elif data_file_name == 'pin_data.npy':
                pin_data = torch.zeros([len(placedb.pin_direct),3])
                for i,net_pin_list in enumerate(placedb.net2pin_map):
                    for pin_id in net_pin_list:
                        parent_node_id = placedb.pin2node_map[pin_id]
                        pin_data[pin_id] = torch.tensor([placedb.pin_offset_x[pin_id],placedb.pin_offset_y[pin_id],int(placedb.pin_direct[pin_id] == b'OUTPUT')])
                np.save(osp.join(netlist_dir,data_file_name),pin_data.squeeze().numpy().astype(np.int))
            elif data_file_name == 'pin_net_cell.npy':#,'cell_clusters.json'
                pin_net_cell = []

                for i,net_pin_list in enumerate(placedb.net2pin_map):
                    for pin_id in net_pin_list:
                        parent_node_id = placedb.pin2node_map[pin_id]
                        pin_net_cell.append([i,parent_node_id])
                pin_net_cell = np.array(pin_net_cell)
                np.save(osp.join(netlist_dir,data_file_name),pin_net_cell)
            elif data_file_name == 'cell_clusters.json':
                
                cells_size = torch.tensor(np.concatenate([placedb.node_x.reshape((-1,1)),placedb.node_y.reshape((-1,1))],axis=1))
                cells_type = torch.zeros([placedb.num_physical_nodes,1])
                fix_node_index_list = list(placedb.rawdb.fixedNodeIndices())
                for i in range(placedb.num_physical_nodes):
                    cells_size[i] = torch.tensor([placedb.node_size_x[i],placedb.node_size_y[i]]) + 1e-5
                    if i in range(placedb.num_movable_nodes):
                        cells_type[i] = 0
                    elif i in fix_node_index_list:
                        cells_type[i] = 1
                    else:
                        cells_type[i] = 2
                
                pin_net_cell = []
                for i,net_pin_list in enumerate(placedb.net2pin_map):
                    for pin_id in net_pin_list:
                        parent_node_id = placedb.pin2node_map[pin_id]
                        pin_net_cell.append([i,parent_node_id])
                pin_net_cell = np.array(pin_net_cell)

                cells = list(pin_net_cell[:, 1])
                nets = list(pin_net_cell[:, 0])

                n_cell = placedb.num_physical_nodes
                n_net = len(placedb.net_names)
                graph = dgl.heterograph({
                    ('cell', 'pins', 'net'): (cells, nets),
                    ('net', 'pinned', 'cell'): (nets, cells),
                    ('cell', 'points-to', 'cell'): ([], []),
                    ('cell', 'pointed-from', 'cell'): ([], []),
                }, num_nodes_dict={'cell': n_cell, 'net': n_net})
                cell_prop_dict = {
                    'size': cells_size,
                    'type': cells_type,
                }
                create_group(graph=graph,
                output_dir=netlist_dir,
                cell_prop_dict=cell_prop_dict,
                keep_cluster_file=False)
            elif data_file_name == "layout_size.json":
                layout_size = [placedb.xh,placedb.yh]
                json_data = json.dumps(layout_size)
                with open(osp.join(netlist_dir,data_file_name),"w") as f:
                    f.write(json_data)
    
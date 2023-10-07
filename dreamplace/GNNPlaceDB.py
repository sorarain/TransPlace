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
from PlaceDB import PlaceDB
import pickle
import json
from thirdparty.TexasCyclone.data.graph.Netlist import Netlist
from thirdparty.TexasCyclone.train.argument import parse_train_args
from thirdparty.TexasCyclone.data.graph import Netlist,Layout,expand_netlist
from copy import deepcopy
import dgl
from tqdm import tqdm
from grouping import create_group

datatypes = {
        'float32' : np.float32,
        'float64' : np.float64
        }

class GNNPlaceDB(PlaceDB):
    """
    @ placement database for hierarchical GNN 
    """
    def __init__(self,
                params,
                dir_name: str,
                save_type: int = 1) -> None:
        super(GNNPlaceDB,self).__init__()
        
        """
        initialize raw place database
        """
        self.read(params)
        self.initialize(params)
        self.params = params#存储params后面legalization会用

        """
        read data from PlaceDB
        """
        print(f'\tLoading {dir_name}')
        self.dataset_name = dir_name
        netlist_pickle_path = f'{dir_name}/netlist.pkl'
        if save_type == 1 and os.path.exists(netlist_pickle_path):
            with open(netlist_pickle_path, 'rb') as fp:
                self.netlist = pickle.load(fp)
        else:
            """
            get cell net num
            """
            n_cell = self.num_physical_nodes
            n_net = len(self.net_names)

            """
            get cell info
            
            cell size, cell degree, cell type
            """
            cells_size = torch.tensor(np.concatenate([self.node_x.reshape((-1,1)),self.node_y.reshape((-1,1))],axis=1))
            cells_degree = torch.zeros([self.num_physical_nodes,1])
            cells_type = torch.zeros([self.num_physical_nodes,1])
            fix_node_index_list = list(self.rawdb.fixedNodeIndices())
            for i in range(self.num_physical_nodes):
                cells_size[i] = torch.tensor([self.node_size_x[i],self.node_size_y[i]])
                if i in range(self.num_movable_nodes):
                    cells_type[i] = 0
                elif i in fix_node_index_list:
                    cells_type[i] = 1
                else:
                    cells_type[i] = 2

            """
            get net info, edge info, pin info 
            inclue 
            net degree, net to cell 
            pin data(0,1 pin offset x,y 2 pin direct 0:in 1:out)
            """
            pin_net_cell = []
            pin_data = torch.zeros([len(self.pin_direct),3])
            nets_degree = torch.zeros([len(self.net_names),1])

            for i,net_pin_list in enumerate(self.net2pin_map):
                nets_degree[i] = len(net_pin_list)
                for pin_id in net_pin_list:
                    parent_node_id = self.pin2node_map[pin_id]
                    cells_degree[parent_node_id] += 1
                    pin_data[pin_id] = torch.tensor([self.pin_offset_x[pin_id],self.pin_offset_y[pin_id],int(self.pin_direct[pin_id] == b'OUTPUT')])
                    pin_net_cell.append([i,parent_node_id])
            pin_net_cell = np.array(pin_net_cell)

            cells = list(pin_net_cell[:, 1])
            nets = list(pin_net_cell[:, 0])

            """
            get cell pos
            """
            if os.path.exists(f'{dir_name}/cell_pos.npy'):
                cells_pos_corner = np.load(f'{dir_name}/cell_pos.npy') #* self.params.scale_factor# - self.params.shift_factor
            else:
                cells_pos_corner = np.zeros(shape=[self.num_physical_nodes, 2], dtype=np.float)
            print(self.xh,self.yh,self.xl,self.yl,params.shift_factor, params.scale_factor)
            print(np.max(cells_pos_corner[cells_type.squeeze() < 1,:]))
            print(np.min(cells_pos_corner[cells_type.squeeze() < 1,:]))
            print(cells_pos_corner.shape)
            print(self.num_physical_nodes)
            print(self.num_nodes)
            print(cells_size.size())
            cells_ref_pos = torch.tensor(cells_pos_corner, dtype=torch.float32) + cells_size / 2
            cells_pos = cells_ref_pos.clone()
            cells_pos[cells_type[:, 0] < 1e-5, :] = torch.nan
            
            """
            split pin info
            pin pos, pin io
            """
            pins_pos = torch.tensor(pin_data[:, [0, 1]], dtype=torch.float32)
            pins_io = torch.tensor(pin_data[:, 2], dtype=torch.float32).unsqueeze(-1)

            """
            same with load_data.py function:netlist_from_numpy_directory
            read cell cluster, layout size
            """
            if os.path.exists(f'{dir_name}/cell_clusters.json'):
                with open(f'{dir_name}/cell_clusters.json') as fp:
                    cell_clusters = json.load(fp)
            else:
                print("Not find cell_cluster.json")
                print("create cell_cluster.json")
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
                output_dir=dir_name,
                cell_prop_dict=cell_prop_dict)
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

            """
            same with load_data.py function:netlist_from_numpy_directory
            construct graph, cell/net/pin feature
            """
            # graph = dgl.heterograph({
            #     ('cell', 'pins', 'net'): (cells, nets),
            #     ('net', 'pinned', 'cell'): (nets, cells),
            #     ('cell', 'points-to', 'cell'): ([], []),
            #     ('cell', 'pointed-from', 'cell'): ([], []),
            # }, num_nodes_dict={'cell': n_cell, 'net': n_net})
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
            # np.save(f'{dir_name}/cell_data.npy',torch.stack([cells_degree.squeeze(),cells_size[:,0],cells_size[:,1],cells_type.squeeze()],dim=1).numpy())
            # np.save(f'{dir_name}/cell_pos.npy',cells_pos_corner)
            # assert os.path.exists(f'{dir_name}/old/cell_pos.npy')
            # assert os.path.exists(f'{dir_name}/old/cell_data.npy')
            # from matplotlib import pyplot as plt
            # plt.scatter(cells_pos_corner[:,0],cells_pos_corner[:,1],s=1)
            # plt.gca().add_patch(plt.Rectangle(xy=(0,0),
            #                 width=self.xh, 
            #                 height=self.yh,
            #                 fill=False, linewidth=10))
            # plt.savefig(f'{dir_name}/test.png')
            # plt.clf()
            # return

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
            self.netlist = Netlist(
                graph=graph,
                layout_size=layout_size,
                hierarchical=cell_clusters is not None and len(cell_clusters),
                cell_clusters=cell_clusters,
                original_netlist=Netlist(
                    graph=deepcopy(graph),
                    layout_size=layout_size, simple=True
                ),
                temp_device="cuda:0"
            )
            if save_type != 0:
                with open(netlist_pickle_path, 'wb+') as fp:
                    pickle.dump(self.netlist, fp)

        
        """
        extra info for placedb 
        原先在PlaceDataCollection中计算得到
        """
        self.num_pins_in_nodes = np.zeros(self.num_nodes)
        for i in range(self.num_physical_nodes):
            self.num_pins_in_nodes[i] = len(self.node2pin_map[i])
        """
        create info for each subnetlist
        """
        dict_netlist = expand_netlist(self.netlist)
        self.sub_netlist_info = {}
        for nid,sub_netlist in tqdm(dict_netlist.items()):
            self.sub_netlist_info[nid] = {}
            cells,nets = sub_netlist.graph.edges(etype='pins')
            pin_pos = sub_netlist.graph.edges['pinned'].data['pos']
            self.sub_netlist_info[nid]['num_physical_nodes'] = sub_netlist.graph.num_nodes(ntype='cell')
            self.sub_netlist_info[nid]['num_nets'] = sub_netlist.graph.num_nodes(ntype='net')
            self.sub_netlist_info[nid]['pin_offset_x'] = pin_pos[:,0]
            self.sub_netlist_info[nid]['pin_offset_y'] = pin_pos[:,1]
            self.sub_netlist_info[nid]['node_size_x'] = sub_netlist.graph.nodes['cell'].data['size'][:,0]
            self.sub_netlist_info[nid]['node_size_y'] = sub_netlist.graph.nodes['cell'].data['size'][:,1]
            self.sub_netlist_info[nid]['cell_type'] = sub_netlist.graph.nodes['cell'].data['type']
            self.sub_netlist_info[nid]['layout_size'] = sub_netlist.layout_size#
            self.sub_netlist_info[nid]['span'] = (sub_netlist.graph.nodes['cell'].data['size'][:,0] * sub_netlist.graph.nodes['cell'].data['size'][:,1]).sum()
            if nid != -1:
                """
                现在sub netlist: layout size
                cell总面积的sqrt(n)*0.5+1 (n为sub netlist中cell数目)
                这样做的目的是让含有cell多的netlist有更大的layout 去调整
                """
                span = (sub_netlist.graph.nodes['cell'].data['size'][:,0] * sub_netlist.graph.nodes['cell'].data['size'][:,1]).sum()
                self.sub_netlist_info[nid]['width'],\
                    self.sub_netlist_info[nid]['height'] = span ** 0.5 * (math.sqrt(sub_netlist.graph.num_nodes(ntype='cell'))*0.01 + 5), \
                                                            span ** 0.5 * (math.sqrt(sub_netlist.graph.num_nodes(ntype='cell'))*0.01 + 5)
                """
                sub netlist 的layout是以0,0为左下角
                """
                self.sub_netlist_info[nid]['xl'], self.sub_netlist_info[nid]['xh'] = 0,self.sub_netlist_info[nid]['width']
                self.sub_netlist_info[nid]['yl'], self.sub_netlist_info[nid]['yh'] = 0,self.sub_netlist_info[nid]['height']
                """
                网格划分策略和DREAMPlace的思想一样，计算desity loss的复杂度为O(b^2log(b^2))b为网格数那么b取sqrt(n) n为cell数
                这样复杂度可以做到O(nlogn)
                """
                self.sub_netlist_info[nid]['num_bins_x'],\
                    self.sub_netlist_info[nid]['num_bins_y'] = int(math.sqrt(sub_netlist.graph.num_nodes(ntype='cell'))*8 + 4),\
                                        int(math.sqrt(sub_netlist.graph.num_nodes(ntype='cell'))*8 + 4)
                """
                这里补成偶数是看了DREAMPlace内部的DCT实现 需要网格数是偶数才行，没有补成偶数
                DREAMPlace原来直接设置num_bins_x num_bins_y均为1024
                """
                if self.sub_netlist_info[nid]['num_bins_x'] % 2 == 1:
                    self.sub_netlist_info[nid]['num_bins_x'] += 1
                if self.sub_netlist_info[nid]['num_bins_y'] % 2 == 1:
                    self.sub_netlist_info[nid]['num_bins_y'] += 1
                self.sub_netlist_info[nid]['num_bins_x'] = min(self.sub_netlist_info[nid]['num_bins_x'],1024)
                self.sub_netlist_info[nid]['num_bins_y'] = min(self.sub_netlist_info[nid]['num_bins_y'],1024)
                # self.netlist.dict_sub_netlist[nid].layout_size = (self.sub_netlist_info[nid]['width'],self.sub_netlist_info[nid]['height'])
            else:
                self.sub_netlist_info[nid]['width'],self.sub_netlist_info[nid]['height'] = self.xh,self.yh
                self.sub_netlist_info[nid]['xl'], self.sub_netlist_info[nid]['xh'] = 0,self.sub_netlist_info[nid]['width']
                self.sub_netlist_info[nid]['yl'], self.sub_netlist_info[nid]['yh'] = 0,self.sub_netlist_info[nid]['height']
                self.sub_netlist_info[nid]['num_bins_x'],\
                    self.sub_netlist_info[nid]['num_bins_y'] = 512,\
                                        512
            self.sub_netlist_info[nid]['bin_size_x'] = self.sub_netlist_info[nid]['width'] / self.sub_netlist_info[nid]['num_bins_x']
            self.sub_netlist_info[nid]['bin_size_y'] = self.sub_netlist_info[nid]['height'] / self.sub_netlist_info[nid]['num_bins_y']

            sub_netlist.layout_size = (self.sub_netlist_info[nid]['width'],self.sub_netlist_info[nid]['height'])
            self.sub_netlist_info[nid]['layout_size'] = sub_netlist.layout_size#
            

            self.sub_netlist_info[nid]['pin2node_map'] = cells
            
            """
            node2pin_map 每个点含有pin_id
            pin2net_map每个pin对应的net_id
            net2pin_map每个net含有的pin_id
            """
            pin_mask_ignore_fixed_macros = torch.zeros_like(cells)
            node2pin_map = [[] for _ in range(sub_netlist.graph.num_nodes(ntype='cell'))]
            pin2net_map = np.zeros(len(cells))
            net2pin_map = [[] for _ in range(sub_netlist.graph.num_nodes(ntype='net'))]
            for pin_id,cell in enumerate(cells):
                net = nets[pin_id]
                node2pin_map[cell].append(pin_id)
                net2pin_map[net].append(pin_id)
                pin2net_map[pin_id] = net
                """
                这里做len(sub_netlist.terminal_indices) < 10判断是发现
                if sub_netlist.cell_prop_dict['type'][cell,0] > 0:做的时候有点慢
                在我们的sub netlist的terminal应该是1个我们指定的最大的cell这个就不用ignore直接过了
                只有原图才会有terminal，会快不少
                """
                if len(sub_netlist.terminal_indices) < 10:
                    # if int(cell) in sub_netlist.terminal_indices:
                    #     pin_mask_ignore_fixed_macros[pin_id] = 1
                    pass
                else:
                    if sub_netlist.graph.nodes['cell'].data['type'][cell,0] > 0:
                        pin_mask_ignore_fixed_macros[pin_id] = 1
            self.sub_netlist_info[nid]['pin2net_map'] = pin2net_map
            
            """
            将node2pin_map pin2net_map net2pin_map展开成1D array *_start_map存储起始下标
            注：*_start_map最后会比原来的flat_*_map多一个元素，值为flat_*_map的长度
            """
            flat_node2pin_map = []
            flat_node2pin_start_map = []
            for cell in range(sub_netlist.graph.num_nodes(ntype='cell')):
                flat_node2pin_start_map.append(len(flat_node2pin_map))
                flat_node2pin_map.extend(node2pin_map[cell])
            flat_node2pin_start_map.append(len(flat_node2pin_map))
            self.sub_netlist_info[nid]['flat_node2pin_map'] = np.array(flat_node2pin_map, dtype=np.int32)
            self.sub_netlist_info[nid]['flat_node2pin_start_map'] = np.array(flat_node2pin_start_map, dtype=np.int32)

            flat_net2pin_map = []
            flat_net2pin_start_map = []
            for net in range(sub_netlist.graph.num_nodes(ntype='net')):
                flat_net2pin_start_map.append(len(flat_net2pin_map))
                flat_net2pin_map.extend(net2pin_map[net])
            flat_net2pin_start_map.append(len(flat_net2pin_map))
            self.sub_netlist_info[nid]['net2pin_map'] = net2pin_map
            self.sub_netlist_info[nid]['flat_net2pin_map'] = np.array(flat_net2pin_map, dtype=np.int32)
            self.sub_netlist_info[nid]['flat_net2pin_start_map'] = np.array(flat_net2pin_start_map, dtype=np.int32)

            self.sub_netlist_info[nid]['net_weights'] = np.ones(sub_netlist.graph.num_nodes(ntype='net'), dtype=np.float32)

            net_degrees = np.array([
                len(net2pin) for net2pin in net2pin_map
            ])
            net_mask = np.logical_and(
                2 <= net_degrees,
                net_degrees < params.ignore_net_degree
            )
            self.sub_netlist_info[nid]['net_mask_ignore_large_degrees'] = net_mask#这里是看了DREAMPlace实现搬过来的，是忽略了度数大的net的hpwl，应该是不太好优化就忽略了
            self.sub_netlist_info[nid]['net_mask_all'] = torch.from_numpy(
                np.ones(sub_netlist.graph.num_nodes(ntype='net'))
            )

            self.sub_netlist_info[nid]['pin_mask_ignore_fixed_macros'] = pin_mask_ignore_fixed_macros

            num_pins_in_nodes = np.zeros(sub_netlist.graph.num_nodes(ntype='cell'))
            for i in range(sub_netlist.graph.num_nodes(ntype='cell')):
                num_pins_in_nodes[i] = len(node2pin_map[i])
            self.sub_netlist_info[nid]['num_pins_in_nodes'] = torch.tensor(num_pins_in_nodes,device=torch.device("cuda:0"))





if __name__ == '__main__':
    args = parse_train_args()
    params = Params.Params()
    params.printWelcome()
    # if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    #     params.printHelp()
    #     exit()
    # elif len(sys.argv) != 2:
    #     logging.error("One input parameters in json format in required")
    #     params.printHelp()
    #     exit()

    # load parameters
    # params.load(sys.argv[1])
    params.load(args.param_json)
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    db = GNNPlaceDB(params,'/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2',1)
    logging.info("initialize GNN placemet database takes %.3f seconds" % (time.time() - tt))
import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))

from typing import Dict, List, Tuple
import dgl
import networkx as nx
import Params
import math
import json
import numpy as np
from time import time

def create_input_graph_file(
    graph:dgl.DGLHeteroGraph,
    input_file_name:str,
    cell_prop_dict = None):

    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')

    if cell_prop_dict is None:
        output_str = f"{num_nets} {num_nodes}\n"
    else:
        output_str = f"{num_nets} {num_nodes} 10\n"

    lines = [[] for _ in range(num_nets)]
    cells,nets = graph.edges(etype='pins')

    for cell,net in zip(cells,nets):
        lines[net].append(int(cell) + 1)
    
    for line in lines:
        for i,cell in enumerate(line):
            output_str += str(int(cell))
            if i != len(line) - 1:
                output_str += ' '
            else:
                output_str += '\n'
    
    if cell_prop_dict is not None:
        for size in cell_prop_dict['size']:
            output_str += str(int(size[0] * size[1])) + '\n'

    with open(input_file_name,'w') as f:
        f.write("".join(output_str))

def overlap(func):
    def fn2(graph,belong,blocks):
        result = func(graph,belong,blocks)
        if not os.path.exists("/root/DREAMPlace/time/overlap_ratio.json"):
            return result
        
        num_nets = graph.num_nodes('net')
        num_nodes = graph.num_nodes('cell')
        overlap_ratio = 0.0

        for i,group_edges in enumerate(result):
            group_graph = nx.Graph(list(group_edges))
            connect_part_set_list = nx.connected_components(group_graph)
            connect_part_lists = [list(node_set) for node_set in connect_part_set_list]
            for connect_part in connect_part_lists:
                tmp_cnt = 0
                for cell in connect_part:
                    if cell > num_nodes:
                        tmp_cnt += 1
                        continue
                overlap_ratio += tmp_cnt * 1.0 / num_nets

        with open("/root/DREAMPlace/time/overlap_ratio.json","r") as f:
            data = json.load(f)
        for k,v in data.items():
            if v == -1:
                data[k] = overlap_ratio
                with open("/root/DREAMPlace/time/overlap_ratio.json","w") as f:
                    f.write(json.dumps(data))
                break
        return result
    return fn2
@overlap
def create_group_graph(
    graph:dgl.DGLHeteroGraph,
    belong,
    blocks: int
):
    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')
    cells,nets = graph.edges(etype='pins')

    nets_id = num_nodes + 3
    net_have_nodes = [{} for _ in range(num_nets)]
    group_graph_edges = [[] for _ in range(blocks)]

    for cell,net in zip(cells,nets):
        cell_belong = int(belong[int(cell)])
        if cell_belong == -1:
            continue
        net_have_nodes[net].setdefault(cell_belong,[])
        net_have_nodes[net][cell_belong].append(int(cell))
    
    for net in range(num_nets):
        for group in net_have_nodes[net]:
            for cell in net_have_nodes[net][group]:
                cell_belong = int(belong[int(cell)])
                if cell_belong == -1:
                    continue
                group_graph_edges[cell_belong].append((int(cell),nets_id))
                group_graph_edges[cell_belong].append((nets_id,int(cell)))
            nets_id += 1
    return group_graph_edges

def create_cluster_json(
    graph:dgl.DGLHeteroGraph,
    input_filename:str,
    output_filename:str,
    blocks: int,
    theta: int,
    cell_prop_dict = None):

    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')
    block_lists = []
    belong = np.ones(num_nodes) * (-1)

    with open(input_filename,'r') as f:
        lines = f.readlines()
    cell_type = cell_prop_dict['type']
    cell_size = cell_prop_dict['size']
    for i,line in enumerate(lines):
        if cell_type[i] > 0 or cell_size[i,0] * cell_size[i,1] > theta:
            continue
        # block_lists[int(line)].append(i)
        belong[i] = int(line)
    
    batch_graph_edges = create_group_graph(graph,belong,blocks)

    for i,group_edges in enumerate(batch_graph_edges):
        group_graph = nx.Graph(list(group_edges))
        connect_part_set_list = nx.connected_components(group_graph)
        connect_part_lists = [list(node_set) for node_set in connect_part_set_list]
        for connect_part in connect_part_lists:
            tmp_result = []
            for cell in connect_part:
                if cell > num_nodes:
                    continue
                tmp_result.append(int(cell))
            block_lists.append(tmp_result)

    print(f"total blocks {len(block_lists)}")

    json_data = json.dumps(block_lists)
    with open(output_filename,"w") as f:
        f.write(json_data)

def check_cluster_json(
    graph:dgl.DGLHeteroGraph,
    json_filename:str
):
    with open(json_filename,'r') as fp:
        cell_cluster = json.load(fp)
    
    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')
    belong = np.ones(num_nodes) * (-1)

    for i,group in enumerate(cell_cluster):
        for cell in group:
            belong[int(cell)] = int(i)
    
    batch_graph_edges = create_group_graph(graph,belong,len(cell_cluster))

    for group_edges in batch_graph_edges:
        group_graph = nx.Graph(list(group_edges))
        if not nx.is_connected(group_graph):
            print("error not connect graph")
            raise 
    
    print("all connect")
    return True


    
    
    
def timer(func):
    def fn2(*args,**kwargs):
        f=time()
        result = func(*args,**kwargs)
        d = time()
        c = d-f
        if os.path.exists("/root/DREAMPlace/time/grouping_time.json"):
            with open("/root/DREAMPlace/time/grouping_time.json","r") as f:
                data = json.load(f)
            for k,v in data.items():
                if v == -1:
                    data[k] = result
                    with open("/root/DREAMPlace/time/grouping_time.json","w") as f:
                        f.write(json.dumps(data))
                    break
        return result
    return fn2
@timer
def create_group(
    graph:dgl.DGLHeteroGraph,
    output_dir: str,
    cell_prop_dict = None,
    keep_cluster_file:bool = False,
    use_kahypar="kahypar"):
    """
    keep_cluster_file是否保留中间结果文件
    """
    blocks = int(math.sqrt(graph.num_nodes('cell')))#math.ceil(graph.num_nodes('cell') / 200000)
    # blocks = math.ceil(graph.num_nodes('cell') / 50000)
    if blocks < 2:
        return
    hmetis_input_filename = os.path.join(output_dir,'graph.input')
    create_input_graph_file(graph,hmetis_input_filename,cell_prop_dict)
    use_kahypar = "mt_strong"
    if use_kahypar == "kahypar":
        cmd = f"./thirdparty/kahypar/build/kahypar/application/KaHyPar -h {hmetis_input_filename} -k {blocks} -e 0.03 -o km1 -m direct -p ./thirdparty/kahypar/config/km1_kKaHyPar_sea20.ini -w true"
        grouping_filename = os.path.join(output_dir,f"graph.input.part{blocks}.epsilon0.03.seed-1.KaHyPar")
    elif use_kahypar == "mt_fast":
        cmd = f"./thirdparty/mt-kahypar/build/mt-kahypar/application/MtKaHyParFast -h {hmetis_input_filename} -k {blocks} -e 0.03 -o km1 -m direct -p ./thirdparty/mt-kahypar/config/fast_preset.ini -t 24"
        grouping_filename = os.path.join(output_dir,f"graph.input.part{blocks}.epsilon0.03.seed0.KaHyPar")
    elif use_kahypar == "mt_strong":
        # cmd = f"./thirdparty/mt-kahypar/build/mt-kahypar/application/MtKaHyParStrong -h {hmetis_input_filename} -k {blocks} -e 0.03 -o km1 -m direct -p ./thirdparty/mt-kahypar/config/strong_preset.ini -t 24"
        cmd = f"./thirdparty/mt-kahypar/build/mt-kahypar/application/MtKaHyPar -h {hmetis_input_filename} --preset-type=default  --instance-type=hypergraph -t 32 -k {blocks} -e 0.03 -o km1 -m direct --write-partition-file=true"
        grouping_filename = os.path.join(output_dir,f"graph.input.part{blocks}.epsilon0.03.seed0.KaHyPar")
    print(cmd)
    a=time()
    os.system(cmd)
    b=time()
    cluster_json_filename = os.path.join(output_dir,"cell_clusters.json")
    theta = 1e9#(cell_prop_dict['size'][:,0] * cell_prop_dict['size'][:,1]).mean()#200
    create_cluster_json(graph,grouping_filename,cluster_json_filename,blocks,theta,cell_prop_dict)
    if not keep_cluster_file:
        os.remove(grouping_filename)
        os.remove(hmetis_input_filename)
    return b-a

if __name__ == '__main__':
    param_json = '/home/xuyanyang/RL/DREAMPlace/test/ispd2015/lefdef/mgc_superblue19.json'
    netlist_name = '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue19'
    params = Params.Params()
    # load parameters
    params.load(param_json)
    from GNNPlaceDB import GNNPlaceDB
    placedb = GNNPlaceDB(params,netlist_name,2)
    print(f"load {netlist_name} netlist")
    print(f"num nodes {placedb.num_nodes}")
    print(f"num_physical_nodes {placedb.num_physical_nodes}")
    # create_group(placedb.netlist.graph,netlist_name,placedb.netlist.cell_prop_dict)
    # check_cluster_json(placedb.netlist.graph,os.path.join(netlist_name,'cell_clusters.json'))
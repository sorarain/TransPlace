import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))
sys.path.append('./thirdparty/TexasCyclone')

from GNNPlace import GNNPlace
from GNNPlaceDB import GNNPlaceDB
from thirdparty.TexasCyclone.train.argument import parse_train_args
import torch
from GenerateData import generate_data
import Params
from GenerateParamJson import create_param_json
import numpy as np


from thirdparty.TexasCyclone.data.utils import check_dir
from thirdparty.TexasCyclone.train.argument import parse_pretrain_args
from thirdparty.TexasCyclone.train.pretrain_ours import pretrain_ours
from typing import List
import PlaceDB
from dreamplace import NonLinearPlace
import logging
import json
import pandas as pd

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='benchmarks'
# PARAM_DIR='test/OurModel_lowepoch_NAG'
# PARAM_DIR='test/dac2012'
# PARAM_DIR='test/ispd2015/lefdef'
# PARAM_DIR='test/ispd2019/lefdef'
# PARAM_DIR = 'result/ispd19-results/ispd19-ours'


def rotate(x, y, theta):
    theta = np.pi / 180
    x_ = x * np.cos(theta) - y * np.sin(theta)
    y_ = x * np.sin(theta) + y * np.cos(theta)
    return x_,y_
def translate(x,y,delta_x,delta_y):
    x_ = x + delta_x
    y_ = y + delta_y
    return x_,y_

def create_time_json_file(file_name:str):
    if os.path.exists(os.path.join("/root/DREAMPlace/time",file_name)):
        return
    data = {}
    with open(os.path.join("/root/DREAMPlace/time",file_name),"w") as f:
        f.write(json.dumps(data))

def add_netlist_name2time_json_file(file_name:str,netlist_name:str):
    with open(os.path.join("/root/DREAMPlace/time",file_name),"r") as f:
        data = json.load(f)
    if netlist_name in data:
        return
    data[netlist_name] = -1
    with open(os.path.join("/root/DREAMPlace/time",file_name),"w") as f:
        f.write(json.dumps(data))

        
def load_placedb(params,
                    netlist_name,
                    name,
                    save_type=1):
    """
    直接将所有placedb都进来存起来
    """
    print(f'Loading {name} data...')
    placedb = GNNPlaceDB(params,netlist_name,save_type)
    print(f"load {netlist_name} netlist")
    print(f"num nodes {placedb.num_nodes}")
    print(f"num_physical_nodes {placedb.num_physical_nodes}")
    return placedb
        
if __name__ == '__main__':
    args = parse_train_args()


    ### This is for ISPD2019: TODO: comment out something in PlaceObj
    # netlist_dir = f'{NETLIST_DIR}/ispd2019/ispd19_test{args.no}'   
    # PARAM_DIR = 'test/OurModel_lowepoch_NAG/ispd2019' 
    # test_param_json = f'{PARAM_DIR}/ispd19_test{args.no}.json'
    # params = Params.Params()
    # params.load(test_param_json)
    # params.lef_input[0] = f"benchmarks/ispd2019/ispd19_test{args.no}/ispd19_test{args.no}.input.lef"
    # params.def_input = f"benchmarks/ispd2019/ispd19_test{args.no}/ispd19_test{args.no}.input.def"
                

    ### This is for DAC:
    netlist_dir = f'{NETLIST_DIR}/dac2012/superblue{args.no}'
    PARAM_DIR = 'test/OurModel_lowepoch_NAG/dac2012'
    test_param_json = f'{PARAM_DIR}/superblue{args.no}.json'
    params = Params.Params()
    params.load(test_param_json)
    params.aux_input = f"benchmarks/dac2012/superblue{args.no}/superblue{args.no}.aux"
    

    block = params.block
    block_x = params.block_x
    block_step_x = params.block_step_x
    block_y = params.block_y
    block_step_y = params.block_step_y
    idx_x = params.idx_x
    idx_y = params.idx_y
    idx = params.idx_theta


    idx_x_ = idx_x - block_x / 2
    idx_y_ = idx_y - block_y / 2
    theta = idx * 360.0 / block 


    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    ############Train
    
    

    
    generate_data(netlist_dir, params,for_test=True)
    
    result = {}

    os.environ["OMP_NUM_THREADS"] = "%d" % (16)


    placedb= load_placedb(params, netlist_dir, 'test',1)
    device = torch.device(args.device)
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'PASS_TYPE': args.pass_type,
        "NUM_LAYERS": args.layers,
    }
    sample_netlist = placedb.netlist
    raw_cell_feats = sample_netlist.graph.nodes['cell'].data['feat'].shape[1]
    raw_net_feats = sample_netlist.graph.nodes['net'].data['feat'].shape[1]
    raw_pin_feats = sample_netlist.graph.edges['pinned'].data['feat'].shape[1]
    placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
    if args.model:
        placer.load_dict(f"./model/{args.model}.pkl",device)
    placer.jump_LGDP = True
    placer.logs.append({'epoch':0})
    
    netlist_name = netlist_dir.split('/')[-1]
    metric_dict = placer.evaluate_place(placedb,placedb.netlist,netlist_name,detail_placement=True,use_tqdm=True)
    placer.logs[-1].update(metric_dict)
    result[netlist_name] = {"eval time":placer.logs[-1][f"{netlist_name} eval_time"]}

    result_ = {}

    
    
    root_save_path = f'./result/{args.name}/{netlist_name}/'
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)

    placedb = PlaceDB.PlaceDB()
    placedb(params)
    netlist_name = netlist_dir.split('/')[-1]
    node_pos = np.load(f'./result/{args.name}/{netlist_name}/{netlist_name}.npy')
    
    ######
    node_pos = np.load(f'./result/{args.name}/{netlist_name}/{netlist_name}.npy')
    x = node_pos[:placedb.num_movable_nodes,0]
    y = node_pos[:placedb.num_movable_nodes,1]
    x_,y_ = rotate(x,y,theta)
    delta_x = idx_x_ * placedb.bin_size_x * block_step_x
    delta_y = idx_y_ * placedb.bin_size_y * block_step_y
    x_,y_ = translate(x_,y_,delta_x,delta_y)
    node_pos[:placedb.num_movable_nodes,0] = x_
    node_pos[:placedb.num_movable_nodes,1] = y_
    np.save(f'./result/{args.name}/{netlist_name}/{netlist_name}_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}_rotate{int(theta)}.npy',node_pos)
    params.__dict__["init_pos_dir"] = f'./result/{args.name}/{netlist_name}/{netlist_name}_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}_rotate{int(theta)}.npy'
    # params.__dict__["save_gp_dir"] = f"./result/{args.name}/{netlist_name}/{netlist_name}_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}_rotate{int(theta)}"
    #####

    
    params.init_pos_dir = f'./result/{args.name}/{netlist_name}/{netlist_name}.npy'
    params.save_gp_dir = root_save_path + "ours"
    
    placer = NonLinearPlace.NonLinearPlace(params, placedb,None)
    metrics = placer(params, placedb)
    if netlist_name not in result:
        result[netlist_name] = {"hpwl":0,"eval time":0}

    result_[netlist_name] = {}
    result_[netlist_name]["hpwl"] = int(metrics[-1].true_hpwl)
    result_[netlist_name]["inference time"] = float(result[netlist_name]["eval time"])
    result_[netlist_name]["dreamplace time"] = float(metrics[-1].optimizer_time)
    result_[netlist_name]["eval time"] = float(metrics[-1].optimizer_time) + float(result[netlist_name]["eval time"])
    result_[netlist_name]["epochs"] = len(metrics)
            
    with open(root_save_path + f"result", "w") as f:
        jsoncontent = json.dumps(result_)
        f.write(jsoncontent)
    
    print(params)
    with open(root_save_path + 'hyperparameter.json', "w") as f:
        json.dump(params.toJson(), f)
                
    print(result_)
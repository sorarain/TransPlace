import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))
sys.path.append('./thirdparty/TexasCyclone')

from GNNPlace import GNNPlace
from GNNPlaceDB import GNNPlaceDB
from GNNPlace import load_placedb
from thirdparty.TexasCyclone.train.argument import parse_train_args
import torch
from GenerateData import generate_data
import Params
from GenerateParamJson import create_param_json


from thirdparty.TexasCyclone.data.utils import check_dir
from thirdparty.TexasCyclone.train.argument import parse_pretrain_args
from thirdparty.TexasCyclone.train.pretrain_ours import pretrain_ours
from typing import List
import PlaceDB
from dreamplace import NonLinearPlace
import logging
import pandas as pd
import Timer
import numpy as np

def add_route(params):
    params.__dict__["routability_opt_flag"] = 1
    params.__dict__["adjust_nctugr_area_flag"] = 0
    params.__dict__["adjust_pin_area_flag"] = 0
    params.__dict__["routability_opt_flag"] = 1
    params.__dict__["max_num_area_adjust"] = 5
    params.__dict__["route_num_bins_x"] = 800
    params.__dict__["route_num_bins_y"] = 415
    params.__dict__["node_area_adjust_overflow"] = 0.20
    params.__dict__["area_adjust_stop_ratio"] = 0.01
    params.__dict__["route_area_adjust_stop_ratio"] = 0.01
    params.__dict__["pin_area_adjust_stop_ratio"] = 0.05
    params.__dict__["unit_horizontal_capacity"] = 50
    params.__dict__["unit_vertical_capacity"] = 58
    params.__dict__["unit_pin_capacity"] = 100
    params.__dict__["max_route_opt_adjust_rate"] = 2.5
    params.__dict__["route_opt_adjust_exponent"] = 2.5
    params.__dict__["pin_stretch_ratio"] = 1.414213562
    params.__dict__["max_pin_opt_adjust_rate"] = 1.5
    params.__dict__["global_place_stages"][0]["iteration"] = 2000
    params.__dict__["global_place_stages"][0]["Llambda_density_weight_iteration"] = 1
    params.__dict__["global_place_stages"][0]["Lsub_iteration"] = 1
    params.__dict__["global_place_stages"][0]["routability_Lsub_iteration"] = 5
    params.__dict__["stop_overflow"] = 0.1
    return params
def rotate(x, y, theta):
    theta = np.pi / 180
    x_ = x * np.cos(theta) - y * np.sin(theta)
    y_ = x * np.sin(theta) + y * np.cos(theta)
    return x_,y_
def translate(x,y,delta_x,delta_y):
    x_ = x + delta_x
    y_ = y + delta_y
    return x_,y_

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    ############Train
    os.environ["OMP_NUM_THREADS"] = "%d" % (16)

    args = parse_train_args()

    result = {}
    benchmark = args.param_json.split('/')[0]
    netlist_name = args.param_json.split('/')[1]
    args.model = args.param_json.split('/')[2]
    if 'ispd' in benchmark:
        param_dir = os.path.join("test","OurModel_lowepoch_NAG",benchmark,netlist_name + ".json")
    else:
        param_dir = os.path.join("test","OurModel_lowepoch_NAG",benchmark,netlist_name + ".json")

    if "iccad" in benchmark:
        placedb = load_placedb([param_dir],["benchmarks/"+benchmark+".ot"+"/"+netlist_name],'test',1)[0]
    else:
        placedb = load_placedb([param_dir],["benchmarks/"+benchmark+"/"+netlist_name],'test',1)[0]
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
    # for placedb,netlist_name in zip(test_placedb_list,test_netlist_names):
    #     netlist_name = netlist_name.split('/')[-1]
    #     _ = placer.evaluate_place(placedb,placedb.netlist,netlist_name,use_tqdm=False)
    # placer.logs = [{'epoch':0}]
    metric_dict = placer.evaluate_place(placedb,placedb.netlist,netlist_name,detail_placement=True,use_tqdm=True)
    placer.logs[-1].update(metric_dict)
    result[netlist_name] = {"gnn time":placer.logs[-1][f"{netlist_name} eval_time"]}
    os.system(f"mkdir -p ./result/{args.name}/{benchmark}/{netlist_name}")
    os.system(f"mv ./result/{args.name}/{netlist_name}/{netlist_name}.npy ./result/{args.name}/{benchmark}/{netlist_name}/{netlist_name}.npy")
    os.system(f"rm -r ./result/{args.name}/{netlist_name}")
    
    params = Params.Params()
    params.load(param_dir)
    timer = None

    if 'ispd' in benchmark or "dac" in benchmark: 
        params.__dict__["adjust_nctugr_area_flag"] = 0
        params.__dict__["adjust_rudy_area_flag"] = 1
        params = add_route(params)
    block_step_x = 10
    block_step_y = 10
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    
    if "iccad" in benchmark:
        if params.timing_opt_flag:
            timer = Timer.Timer()
            timer(params, placedb)
            # This must be done to explicitly execute the parser builders.
            # The parsers in OpenTimer are all in lazy mode.
            timer.update_timing()
    if "block_x" in params.__dict__:
        block_x = params.__dict__["block_x"]
    else:
        block_x = 2
    if "block_y" in params.__dict__:
        block_y = params.__dict__["block_y"]
    else:
        block_y = 2
    if "idx_theta" in params.__dict__:
        idx = params.idx_theta
    elif "idx" in params.__dict__:
        idx = params.idx
    else:
        idx = 0
    if "idx_x" not in params.__dict__:
        idx_x = block_x / 2
    else:
        idx_x = params.__dict__["idx_x"]
    if "idx_y" not in params.__dict__:
        idx_y = block_y / 2
    else:
        idx_y = params.__dict__["idx_y"]
    if "block" not in params.__dict__:
        block = 5
    else:
        block = params.block
    idx_x_ = idx_x - block_x / 2
    idx_y_ = idx_y - block_y / 2
    theta = idx * 360.0 / block
    params.__dict__["save_gp_dir"] = f"./result/{args.name}/{benchmark}/{netlist_name}/{netlist_name}"
    params.__dict__["init_pos_dir"] = f'./result/{args.name}/{benchmark}/{netlist_name}/{netlist_name}_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}_rotate{int(theta)}.npy'
    node_pos = np.load(f'./result/{args.name}/{benchmark}/{netlist_name}/{netlist_name}.npy')
    x = node_pos[:placedb.num_movable_nodes,0]
    y = node_pos[:placedb.num_movable_nodes,1]
    x_,y_ = rotate(x,y,theta)
    delta_x = idx_x_ * placedb.bin_size_x * block_step_x
    delta_y = idx_y_ * placedb.bin_size_y * block_step_y
    x_,y_ = translate(x_,y_,delta_x,delta_y)
    node_pos[:placedb.num_movable_nodes,0] = x_
    node_pos[:placedb.num_movable_nodes,1] = y_
    np.save(f'./result/{args.name}/{benchmark}/{netlist_name}/{netlist_name}_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}_rotate{int(theta)}.npy',node_pos)
    placer = NonLinearPlace.NonLinearPlace(params, placedb,timer)
    metrics = placer(params, placedb)
    result[netlist_name]["hpwl"] = metrics[-1].true_hpwl
    result[netlist_name]["dreamplace time"] = metrics[-1].optimizer_time
    result[netlist_name]["eval time"] = metrics[-1].optimizer_time + result[netlist_name]["gnn time"]
    result[netlist_name]["epochs"] = len(metrics)
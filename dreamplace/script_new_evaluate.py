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
PARAM_DIR='test/OurModel_lowepoch_NAG'
# PARAM_DIR='test/dac2012'
# PARAM_DIR='test/ispd2015/lefdef'
# PARAM_DIR='test/ispd2005'

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

def generate_data_list(netlist_dir_list:List[str],param_dir_list:List[str]):
    # create_time_json_file("cellflow_time.json")
    # create_time_json_file("grouping_time.json")
    # create_time_json_file("overlap_ratio.json")

    
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):

        netlist_name = netlist_dir.split("/")[-1]
        # add_netlist_name2time_json_file("cellflow_time.json",netlist_name)
        # add_netlist_name2time_json_file("grouping_time.json",netlist_name)
        # add_netlist_name2time_json_file("overlap_ratio.json",netlist_name)
        
        if not os.path.exists(param_dir):
            os.system(f"mkdir -p {os.path.dirname(param_dir)}")
            create_param_json(netlist_dir,param_dir,ourmodel=True)
        params = Params.Params()
        params.load(param_dir)
        generate_data(netlist_dir,params,for_test=True)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    test_param_json_list = [
        # 'test/dac2012/superblue2.json'
        # 'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
        # f'{PARAM_DIR}/mgc_fft_1/mgc_fft_1.json',
        # f'{PARAM_DIR}/ispd19_test1/ispd19_test1.json',
        # f'{PARAM_DIR}/mgc_superblue19/mgc_superblue19.json',
        # f'{PARAM_DIR}/mgc_des_perf_1.json',
        # f'{PARAM_DIR}/mgc_fft_1.json',
        # f'{PARAM_DIR}/mgc_fft_2.json',
        # f'{PARAM_DIR}/mgc_fft_a.json',
        # f'{PARAM_DIR}/mgc_fft_b.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_1.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_2.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_a.json',
        # f'{PARAM_DIR}/mgc_superblue12.json',
        # f'{PARAM_DIR}/mgc_superblue14.json',
        # f'{PARAM_DIR}/mgc_superblue19.json',
        # f'{PARAM_DIR}/ispd19_test1.json',
        # f'{PARAM_DIR}/ispd19_test2.json',
        # f'{PARAM_DIR}/ispd19_test3.json',
        # f'{PARAM_DIR}/ispd19_test4.json',
        # f'{PARAM_DIR}/ispd19_test6.json',
        # f'{PARAM_DIR}/ispd19_test7.json',
        # f'{PARAM_DIR}/ispd19_test8.json',
        # f'{PARAM_DIR}/ispd19_test9.json',
        # f'{PARAM_DIR}/ispd19_test10.json',
        f'{PARAM_DIR}/superblue1.json',
        f'{PARAM_DIR}/superblue2.json',
        f'{PARAM_DIR}/superblue3.json',
        f'{PARAM_DIR}/superblue6.json',
        f'{PARAM_DIR}/superblue7.json',
        f'{PARAM_DIR}/superblue9.json',
        f'{PARAM_DIR}/superblue11.json',
        f'{PARAM_DIR}/superblue12.json',
        f'{PARAM_DIR}/superblue14.json',
        f'{PARAM_DIR}/superblue16.json',
        f'{PARAM_DIR}/superblue18.json',
        # f'{PARAM_DIR}/superblue19.json',
        # f'{PARAM_DIR}/adaptec1.json',
        # f'{PARAM_DIR}/adaptec2.json',
        # f'{PARAM_DIR}/adaptec3.json',
        # f'{PARAM_DIR}/adaptec4.json',
        # f'{PARAM_DIR}/bigblue1.json',
        # f'{PARAM_DIR}/bigblue2.json',
        # f'{PARAM_DIR}/bigblue3.json',
        # f'{PARAM_DIR}/bigblue4.json',
    ]
    test_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        # f'{NETLIST_DIR}/ispd2015/mgc_des_perf_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue12',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue14',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test1',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test2',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test3',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test4',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test6',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test7',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test8',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test9',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test10',
        f'{NETLIST_DIR}/dac2012/superblue1',
        f'{NETLIST_DIR}/dac2012/superblue2',
        f'{NETLIST_DIR}/dac2012/superblue3',
        f'{NETLIST_DIR}/dac2012/superblue6',
        f'{NETLIST_DIR}/dac2012/superblue7',
        f'{NETLIST_DIR}/dac2012/superblue9',
        f'{NETLIST_DIR}/dac2012/superblue11',
        f'{NETLIST_DIR}/dac2012/superblue12',
        f'{NETLIST_DIR}/dac2012/superblue14',
        f'{NETLIST_DIR}/dac2012/superblue16',
        f'{NETLIST_DIR}/dac2012/superblue18',
        # f'{NETLIST_DIR}/dac2012/superblue19',
        # f'{NETLIST_DIR}/ispd2005/adaptec1',
        # f'{NETLIST_DIR}/ispd2005/adaptec2',
        # f'{NETLIST_DIR}/ispd2005/adaptec3',
        # f'{NETLIST_DIR}/ispd2005/adaptec4',
        # f'{NETLIST_DIR}/ispd2005/bigblue1',
        # f'{NETLIST_DIR}/ispd2005/bigblue2',
        # f'{NETLIST_DIR}/ispd2005/bigblue3',
        # f'{NETLIST_DIR}/ispd2005/bigblue4',
    ]
    ############Train
    args = parse_train_args()
    test_netlist_names = [f'{NETLIST_DIR}/dac2012/superblue{args.no}']
    test_param_json_list = [f'{PARAM_DIR}/superblue{args.no}.json']
    generate_data_list(test_netlist_names,test_param_json_list)
    result = {}

    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    jump_model_inference = True
    jump_dreamplace = False
    if not jump_model_inference:
        test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test',1)
        device = torch.device(args.device)
        config = {
            'DEVICE': device,
            'CELL_FEATS': args.cell_feats,
            'NET_FEATS': args.net_feats,
            'PIN_FEATS': args.pin_feats,
            'PASS_TYPE': args.pass_type,
            "NUM_LAYERS": args.layers,
        }
        sample_netlist = test_placedb_list[0].netlist
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
        for placedb,netlist_name in zip(test_placedb_list,test_netlist_names):
            netlist_name = netlist_name.split('/')[-1]
            metric_dict = placer.evaluate_place(placedb,placedb.netlist,netlist_name,detail_placement=True,use_tqdm=True)
            placer.logs[-1].update(metric_dict)
            result[netlist_name] = {"eval time":placer.logs[-1][f"{netlist_name} eval_time"]}
            # result[netlist_name]['hpwl'] = metric_dict['hpwl']
    
    result_ = {}
    if not jump_dreamplace:
        block = 4
        block_x = 2
        block_step_x = 10
        block_y = 2
        block_step_y = 10
        for idx_x in range(block_x):
            for idx_y in range(block_y):
                for idx in range(block):
                    idx_x_ = idx_x - block_x / 2
                    idx_y_ = idx_y - block_y / 2
                    theta = idx * 360.0 / block
                    for netlist_dir,param_dir in zip(test_netlist_names,test_param_json_list):
                        if not os.path.exists(param_dir):
                            create_param_json(netlist_dir,param_dir)
                        params = Params.Params()
                        params.load(param_dir)
                        # params.__dict__["global_place_flag"] = 1
                        # params.__dict__["global_place_stages"][0]["iteration"] = 0
                        # params.__dict__["legalize_flag"] = 0
                        # params.__dict__["detailed_place_flag"] = 0
                        params.__dict__["adjust_nctugr_area_flag"] = 0
                        params.__dict__["adjust_rudy_area_flag"] = 1
                        placedb = PlaceDB.PlaceDB()
                        placedb(params)
                        netlist_name = netlist_dir.split('/')[-1]
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
                        params.__dict__["save_gp_dir"] = f"./result/{args.name}/{netlist_name}/{netlist_name}_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}_rotate{int(theta)}"
                        # params.__dict__["RePlAce_UPPER_PCOF"] = 1.07
                        placer = NonLinearPlace.NonLinearPlace(params, placedb,None)
                        metrics = placer(params, placedb)
                        if netlist_name not in result:
                            result[netlist_name] = {"hpwl":0,"eval time":0}
                        # result[netlist_name]["hpwl"] = metrics[-1].true_hpwl
                        # result[netlist_name]["inference time"] = result[netlist_name]["eval time"]
                        # result[netlist_name]["dreamplace time"] = metrics[-1].optimizer_time
                        # result[netlist_name]["eval time"] += metrics[-1].optimizer_time
                        # result[netlist_name]["epochs"] = len(metrics)
                        result_[netlist_name + f"_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}" + f"_rotate{int(theta)}"] = {}
                        result_[netlist_name + f"_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}" + f"_rotate{int(theta)}"]["hpwl"] = int(metrics[-1].true_hpwl)
                        result_[netlist_name + f"_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}" + f"_rotate{int(theta)}"]["inference time"] = float(result[netlist_name]["eval time"])
                        result_[netlist_name + f"_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}" + f"_rotate{int(theta)}"]["dreamplace time"] = float(metrics[-1].optimizer_time)
                        result_[netlist_name + f"_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}" + f"_rotate{int(theta)}"]["eval time"] = float(metrics[-1].optimizer_time) + float(result[netlist_name]["eval time"])
                        result_[netlist_name + f"_translate_x{str(idx_x_ * block_step_x)}_y{str(idx_y_ * block_step_y)}" + f"_rotate{int(theta)}"]["epochs"] = len(metrics)
                    with open("./exam_new.json","w") as f:
                        jsoncontent = json.dumps(result_)
                        f.write(jsoncontent)
    print(result_)
    # with open(f"./result/{args.name}/result_modelispd2015_ispd2019_ourmodel.json","w") as f:
    #     for k,v in result.items():
    #         for k_,v_ in v.items():
    #             v[k_] = float(v_)
    #         result[k] = v
    #     f.write(json.dumps(result))
    #     keys = list(result.keys())
    #     result_name_list = list(result[keys[0]].keys())
    #     df = pd.DataFrame()
    #     df['netlist'] = keys
    #     for result_name in result_name_list:
    #         tmp_result = []
    #         for key in keys:
    #             if not result_name in result[key]:
    #                 tmp_result.append(-1)
    #             else:
    #                 tmp_result.append(float(result[key][result_name]))
    #         df[result_name] = tmp_result
    #     df.to_excel(f"./result/{args.name}/result_modelispd2015_ispd2019_ourmodel.xlsx",index=False)
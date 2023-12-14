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

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='benchmarks'
PARAM_DIR='test/ispd2015/lefdef'
# PARAM_DIR='test/DREAMPlace_adam_long'
# PARAM_DIR='test/dac2012'
# PARAM_DIR='test/ispd2005'

def generate_param_list(netlist_dir_list:List[str],param_dir_list:List[str]):
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):
        if not os.path.exists(param_dir):
            os.system(f"mkdir -p {os.path.dirname(param_dir)}")
            create_param_json(netlist_dir,param_dir)
        params = Params.Params()
        params.load(param_dir)

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
        # 'test/DREAMPlace/ispd19_test1/ispd19_test1.json'
        # f'{PARAM_DIR}/mgc_des_perf_1.json',
        # f'{PARAM_DIR}/mgc_des_perf_a.json',
        # f'{PARAM_DIR}/mgc_des_perf_b.json',
        # f'{PARAM_DIR}/mgc_edit_dist_a.json',
        # f'{PARAM_DIR}/mgc_fft_1.json',
        # f'{PARAM_DIR}/mgc_fft_2.json',
        # f'{PARAM_DIR}/mgc_fft_a.json',
        # f'{PARAM_DIR}/mgc_fft_b.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_1.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_2.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_a.json',
        # f'{PARAM_DIR}/mgc_matrix_mult_b.json',#wrong
        # f'{PARAM_DIR}/mgc_matrix_mult_c.json',#wrong
        # f'{PARAM_DIR}/mgc_pci_bridge32_a.json',
        # f'{PARAM_DIR}/mgc_pci_bridge32_b.json',
        # f'{PARAM_DIR}/mgc_superblue11_a.json',
        f'{PARAM_DIR}/mgc_superblue12.json',
        # f'{PARAM_DIR}/mgc_superblue14.json',
        # f'{PARAM_DIR}/mgc_superblue16_a.json',
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
        # f'{PARAM_DIR}/superblue1.json',
        # f'{PARAM_DIR}/superblue2.json',
        # f'{PARAM_DIR}/superblue3.json',
        # f'{PARAM_DIR}/superblue6.json',
        # f'{PARAM_DIR}/superblue7.json',
        # f'{PARAM_DIR}/superblue9.json',
        # f'{PARAM_DIR}/superblue11.json',
        # f'{PARAM_DIR}/superblue12.json',
        # f'{PARAM_DIR}/superblue14.json',
        # f'{PARAM_DIR}/superblue16.json',
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
        # f'{NETLIST_DIR}/ispd2015/mgc_des_perf_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_des_perf_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_edit_dist_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_c',
        # f'{NETLIST_DIR}/ispd2015/mgc_pci_bridge32_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_pci_bridge32_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue11_a',
        f'{NETLIST_DIR}/ispd2015/mgc_superblue12',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue14',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue16_a',
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
        # f'{NETLIST_DIR}/dac2012/superblue1',
        # f'{NETLIST_DIR}/dac2012/superblue2',
        # f'{NETLIST_DIR}/dac2012/superblue3',
        # f'{NETLIST_DIR}/dac2012/superblue6',
        # f'{NETLIST_DIR}/dac2012/superblue7',
        # f'{NETLIST_DIR}/dac2012/superblue9',
        # f'{NETLIST_DIR}/dac2012/superblue11',
        # f'{NETLIST_DIR}/dac2012/superblue12',
        # f'{NETLIST_DIR}/dac2012/superblue14',
        # f'{NETLIST_DIR}/dac2012/superblue16',
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
    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    generate_param_list(test_netlist_names,test_param_json_list)

    result = {}
    for netlist_dir,param_dir in zip(test_netlist_names,test_param_json_list):
        if not os.path.exists(param_dir):
            create_param_json(param_dir)
        params = Params.Params()
        params.load(param_dir)
        params.__dict__["adjust_nctugr_area_flag"] = 0
        params.__dict__["adjust_rudy_area_flag"] = 1
        params = add_route(params)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        netlist_name = netlist_dir.split('/')[-1]
        os.system(f"mkdir -p ./result/test/{netlist_name}")
        params.__dict__["save_gp_dir"] = f"./result/test/{netlist_name}/DREAMPlace_adam{netlist_name}"
        assert not hasattr(params,'init_pos_dir')
        placer = NonLinearPlace.NonLinearPlace(params, placedb,None)
        metrics = placer(params, placedb)
        result[netlist_name] ={}
        result[netlist_name]["hpwl"] = metrics[-1].true_hpwl
        result[netlist_name]["eval time"] = metrics[-1].optimizer_time
        result[netlist_name]["epochs"] = len(metrics)
        # os.system(f"./NCTUgr ICCAD {os.path.join(netlist_dir,netlist_name + '.aux')} {os.path.join('./result','__DREAMPlace_adam',netlist_name,'DREAMPlace_adam' + netlist_name + '.gp.pl')} DAC12.set {os.path.join('./result','__DREAMPlace_adam',netlist_name,'DREAMPlace_adam' + netlist_name + '.gr')}")
    print(result)
    # keys = list(result.keys())
    # result_name_list = list(result[keys[0]].keys())
    # df = pd.DataFrame()
    # df['netlist'] = keys
    # for result_name in result_name_list:
    #     tmp_result = []
    #     for key in keys:
    #         if not result_name in result[key]:
    #             tmp_result.append(-1)
    #         else:
    #             tmp_result.append(float(result[key][result_name]))
    #     df[result_name] = tmp_result
    # df.to_excel(f"./result/__DREAMPlace_adam/result_ispd2015_NAG_2080Ti.xlsx",index=False)
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

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
NETLIST_DIR='./benchmarks'
PARAM_DIR='test/ispd2019'
import time

def generate_data_list(netlist_dir_list:List[str],param_dir_list:List[str],save_type=1):
    for netlist_dir,param_dir in zip(netlist_dir_list,param_dir_list):
        if not os.path.exists(param_dir):
            create_param_json(netlist_dir,param_dir)
        params = Params.Params()
        params.load(param_dir)
        generate_data(netlist_dir,params,save_type,for_test=False)
        
if __name__ == '__main__':
    train_param_json_list = [
        # 'test/dac2012/superblue2.json'
        # 'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue19.json',
        # 'test/ispd2019/lefdef/ispd19_test1.json',
        # 'test/ispd2019/lefdef/ispd19_test2.json',
        # 'test/ispd2019/lefdef/ispd19_test3.json',
        f'{PARAM_DIR}/superblue11.json',
        f'{PARAM_DIR}/superblue12.json',
        f'{PARAM_DIR}/superblue14.json',
        f'{PARAM_DIR}/superblue16.json',
        f'{PARAM_DIR}/superblue19.json',
        # 'test/ispd2005/adaptec1.json',
        # 'test/ispd2005/adaptec2.json',
        # 'test/ispd2005/adaptec3.json',
        # 'test/ispd2005/adaptec4.json',
    ]
    train_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test1',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test2',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test3',
        f'{NETLIST_DIR}/dac2012/superblue11',
        f'{NETLIST_DIR}/dac2012/superblue12',
        f'{NETLIST_DIR}/dac2012/superblue14',
        f'{NETLIST_DIR}/dac2012/superblue16',
        f'{NETLIST_DIR}/dac2012/superblue19',
        # f'{NETLIST_DIR}/ispd2005/adaptec1',
        # f'{NETLIST_DIR}/ispd2005/adaptec2',
        # f'{NETLIST_DIR}/ispd2005/adaptec3',
        # f'{NETLIST_DIR}/ispd2005/adaptec4',
    ]
    valid_param_json_list = [
        # 'test/dac2012/superblue2.json'
        # 'test/ispd2015/lefdef/mgc_fft_1.json',
        # 'test/ispd2015/lefdef/mgc_fft_2.json',
        # 'test/ispd2015/lefdef/mgc_fft_a.json',
        # 'test/ispd2015/lefdef/mgc_fft_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue14.json',
        # 'test/ispd2019/lefdef/ispd19_test1.json',
        # 'test/ispd2019/lefdef/ispd19_test2.json',
        # 'test/ispd2019/lefdef/ispd19_test3.json',
        f'{PARAM_DIR}/superblue11.json',
        f'{PARAM_DIR}/superblue12.json',
        f'{PARAM_DIR}/superblue14.json',
        f'{PARAM_DIR}/superblue16.json',
        f'{PARAM_DIR}/superblue19.json',
        # 'test/ispd2005/adaptec1.json',
        # 'test/ispd2005/adaptec2.json',
        # 'test/ispd2005/adaptec3.json',
        # 'test/ispd2005/adaptec4.json',
    ]
    valid_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue14',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test1',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test2',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test3',
        f'{NETLIST_DIR}/dac2012/superblue11',
        f'{NETLIST_DIR}/dac2012/superblue12',
        f'{NETLIST_DIR}/dac2012/superblue14',
        f'{NETLIST_DIR}/dac2012/superblue16',
        f'{NETLIST_DIR}/dac2012/superblue19',
        # f'{NETLIST_DIR}/ispd2005/adaptec1',
        # f'{NETLIST_DIR}/ispd2005/adaptec2',
        # f'{NETLIST_DIR}/ispd2005/adaptec3',
        # f'{NETLIST_DIR}/ispd2005/adaptec4',
    ]
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
        # 'test/ispd2015/lefdef/mgc_superblue12.json',
        # 'test/ispd2019/lefdef/ispd19_test1.json',
        # 'test/ispd2019/lefdef/ispd19_test2.json',
        # 'test/ispd2019/lefdef/ispd19_test3.json',
        f'{PARAM_DIR}/superblue11.json',
        f'{PARAM_DIR}/superblue12.json',
        f'{PARAM_DIR}/superblue14.json',
        f'{PARAM_DIR}/superblue16.json',
        f'{PARAM_DIR}/superblue19.json',
        # 'test/ispd2005/adaptec1.json',
        # 'test/ispd2005/adaptec2.json',
        # 'test/ispd2005/adaptec3.json',
        # 'test/ispd2005/adaptec4.json',
    ]
    test_netlist_names = [
        # f'{NETLIST_DIR}/dac2012/superblue2'
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_fft_b',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_1',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_2',
        # f'{NETLIST_DIR}/ispd2015/mgc_matrix_mult_a',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue19',
        # f'{NETLIST_DIR}/ispd2015/mgc_superblue12',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test1',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test2',
        # f'{NETLIST_DIR}/ispd2019/ispd19_test3',
        f'{NETLIST_DIR}/dac2012/superblue14',
        f'{NETLIST_DIR}/dac2012/superblue16',
        f'{NETLIST_DIR}/dac2012/superblue19',
        # f'{NETLIST_DIR}/ispd2005/adaptec1',
        # f'{NETLIST_DIR}/ispd2005/adaptec2',
        # f'{NETLIST_DIR}/ispd2005/adaptec3',
        # f'{NETLIST_DIR}/ispd2005/adaptec4',
        
    ]
    generate_data_list(train_netlist_names,train_param_json_list,1)
    generate_data_list(valid_netlist_names,valid_param_json_list,1)
    generate_data_list(test_netlist_names,test_param_json_list,1)
    ############Pretrain
    check_dir(LOG_DIR)
    check_dir(FIG_DIR)
    check_dir(MODEL_DIR)
    args = parse_pretrain_args()
    args.lr = 5e-5
    name = args.name
    args.name = name
    b=time.time()
    pretrain_ours(
        args=args,
        train_datasets=train_netlist_names,
        valid_datasets=valid_netlist_names,
        test_datasets=test_netlist_names,
        log_dir=LOG_DIR,
        fig_dir=FIG_DIR,
        model_dir=MODEL_DIR
    )
    a=time.time()
    # with open("./train_time_ispd2019.txt","w") as f:
    #     f.write(f"pretrain time{a-b}")
    ############Pretrain

    ############Train
    # train_placedb_list =  load_placedb(train_param_json_list,train_netlist_names,'train',2)
    # valid_placedb_list = load_placedb(valid_param_json_list,valid_netlist_names,'valid',1)
    # test_placedb_list = load_placedb(test_param_json_list,test_netlist_names,'test',1)

    # os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    # device = torch.device(args.device)
    # config = {
    #     'DEVICE': device,
    #     'CELL_FEATS': args.cell_feats,
    #     'NET_FEATS': args.net_feats,
    #     'PIN_FEATS': args.pin_feats,
    #     'PASS_TYPE': args.pass_type,
    #     'NUM_LAYERS':args.layers,
    # }
    # sample_netlist = train_placedb_list[0].netlist
    # raw_cell_feats = sample_netlist.graph.nodes['cell'].data['feat'].shape[1]
    # raw_net_feats = sample_netlist.graph.nodes['net'].data['feat'].shape[1]
    # raw_pin_feats = sample_netlist.graph.edges['pinned'].data['feat'].shape[1]
    # args.lr = 1e-6
    # args.epochs=10
    # args.name = "train_"+name
    # args.model = "pretrain_"+name
    # placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
    # # placer.load_dict(f"./model/{args.model}.pkl",device)
    # # placer.load_dict(f"./model/pretrain_cellflow_kahypar_cellprop.pkl",device)
    # # if args.model:
    # #     placer.load_dict(f"./model/{args.model}.pkl",device)
    # if os.path.exists(os.path.join(MODEL_DIR,f"{args.model}.pkl")):
    #     placer.load_dict(os.path.join(MODEL_DIR,f"{args.model}.pkl"),device)
    # bb=time.time()
    # placer.train_epochs(args,train_placedb_list=train_placedb_list,
    #                     train_netlist_names=train_netlist_names,
    #                     valid_placedb_list=valid_placedb_list,
    #                     valid_netlist_names=valid_netlist_names,
    #                     test_placedb_list=test_placedb_list,
    #                     test_netlist_names=test_netlist_names)
    # aa=time.time()
    # print(f"total time {aa-bb+a-b}\npretrain time{b-a}\ntrain time{aa-bb}")
    # # with open("./train_time_ispd2019.txt","w") as f:
    # #     f.write(f"total time {aa-bb+a-b}\npretrain time{b-a}\ntrain time{aa-bb}")
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
import json

if __name__ == '__main__':
    train_param_json_list = [
        # 'test/dac2012/superblue2.json'
        # 'test/ispd2015/lefdef/mgc_des_perf_1.json',#ok finish repairing
        # 'test/ispd2015/lefdef/mgc_des_perf_b.json',
        # 'test/ispd2015/lefdef/mgc_edit_dist_a.json',

        'test/ispd2015/lefdef/mgc_fft_1.json',
        'test/ispd2015/lefdef/mgc_fft_2.json',
        'test/ispd2015/lefdef/mgc_fft_a.json',
        'test/ispd2015/lefdef/mgc_fft_b.json',

        'test/ispd2015/lefdef/mgc_matrix_mult_1.json',
        'test/ispd2015/lefdef/mgc_matrix_mult_2.json',
        'test/ispd2015/lefdef/mgc_matrix_mult_a.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_b.json',
        # 'test/ispd2015/lefdef/mgc_matrix_mult_c.json',

        # 'test/ispd2015/lefdef/mgc_pci_bridge32_a.json',
        # 'test/ispd2015/lefdef/mgc_pci_bridge32_b.json',

        # 'test/ispd2015/lefdef/mgc_superblue11_a.json',
        # 'test/ispd2015/lefdef/mgc_superblue12.json',
        # 'test/ispd2015/lefdef/mgc_superblue14.json',#xxx
        # 'test/ispd2015/lefdef/mgc_superblue16_a.json',
        'test/ispd2015/lefdef/mgc_superblue19.json',
    ]
    train_netlist_names = [
        # '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_des_perf_1',#ok finish repairing
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_des_perf_b',#region
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_edit_dist_a',#region

        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_1',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_2',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_a',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_b',

        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_1',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_2',
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_a',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_b',#region
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_matrix_mult_c',#region

        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_pci_bridge32_a',#region
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_pci_bridge32_b',#region

        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue11_a',#region
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue12',
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue14',#xxx
        # '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue16_a',#region
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue19',

    ]
    valid_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json'
    ]
    valid_netlist_names = [
        # '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_1'
    ]
    test_param_json_list = [
        # 'test/dac2012/superblue2.json'
        'test/ispd2015/lefdef/mgc_fft_1.json'
    ]
    test_netlist_names = [
        # '/home/xuyanyang/RL/Placement-datasets/dac2012/superblue2'
        '/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_fft_1'
    ]
    train_placedb_list =  load_placedb(train_param_json_list,train_netlist_names,'test',1)
    

    args = parse_train_args()
    os.environ["OMP_NUM_THREADS"] = "%d" % (16)
    device = torch.device(args.device)
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'PASS_TYPE': args.pass_type,
    }
    sample_netlist = train_placedb_list[0].netlist
    raw_cell_feats = sample_netlist.graph.nodes['cell'].data['feat'].shape[1]
    raw_net_feats = sample_netlist.graph.nodes['net'].data['feat'].shape[1]
    raw_pin_feats = sample_netlist.graph.edges['pinned'].data['feat'].shape[1]
    placer = GNNPlace(raw_cell_feats, raw_net_feats, raw_pin_feats, config,args)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/pre-default-lr1e-5-mgc_fft_1.pkl',device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/default.pkl',device)
    # placer.load_dict('/home/xuyanyang/RL/DREAMPlace/model/train-default-3layers.pkl',device)
    if args.model:
        placer.load_dict(f"./model/{args.model}.pkl",device)
    placer.evaluate_model(train_placedb_list,train_netlist_names,finetuning=0,name='test')
    print(placer.logs)
    # if not os.path.exists(f"./result/{args.name}"):
    #     os.mkdir(f"./result/{args.name}")
    # with open(f"./result/{args.name}/result.json","w") as f:
    #     json.dump(placer.logs,f)
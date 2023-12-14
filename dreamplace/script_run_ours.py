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
import re
import numpy as np

benchmarks_list = {
    'ispd2015':[
        'mgc_des_perf_1', 
'mgc_des_perf_a', 
'mgc_des_perf_b', 
'mgc_edit_dist_a', 
'mgc_fft_1', 
'mgc_fft_2', 
'mgc_fft_a', 
# 'mgc_fft_b', 
'mgc_matrix_mult_1', 
'mgc_matrix_mult_2', 
'mgc_matrix_mult_a', 
# 'mgc_matrix_mult_b', #wrong
# 'mgc_matrix_mult_c', #wrong
'mgc_pci_bridge32_a', 
'mgc_pci_bridge32_b', 
'mgc_superblue11_a',
'mgc_superblue12', 
'mgc_superblue14', 
'mgc_superblue16_a',
'mgc_superblue19',
],
'dac2012':[
        'superblue2',
        'superblue3',
        'superblue6',
        'superblue7',
        'superblue9',
        'superblue11',
        'superblue12',
        'superblue14',
        'superblue16',
        'superblue19',
],
'ispd2019':[
    'ispd19_test1',
        'ispd19_test2',
        'ispd19_test3',
        'ispd19_test4',
        'ispd19_test6',
        'ispd19_test7',
        'ispd19_test8',
        'ispd19_test9',
        'ispd19_test10',
],
'iccad2015':[
    'superblue1',
        'superblue3',
        'superblue4',
        'superblue5',
        'superblue7',
        'superblue10',
        'superblue16',
        'superblue18',
]
}

args = parse_train_args()

for benchmarks_name,netlist_name_list in benchmarks_list.items():
    if(len(netlist_name_list) == 0):
        continue
    for netlist_name in netlist_name_list:
        os.system(f"mkdir -p ./result/{args.name}/{benchmarks_name}/{netlist_name}")
        os.system(f"python dreamplace/run_ours_time.py --param_json {benchmarks_name+'/' +netlist_name+'/'+args.model+'/'} --name {args.name}")


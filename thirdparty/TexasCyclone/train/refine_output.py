import argparse
import numpy as np
import torch
import json
import os
from typing import List, Dict, Any
from time import time
from tqdm import tqdm

from data.load_data import netlist_from_numpy_directory, layout_from_netlist_cell_pos
from data.utils import set_seed
from train.refine import refined_layout_pos, refined_force_pos


def refine_output(
        refine_datasets: List[str],
        refine_tokens: List[str],
        seed=0, use_tqdm=False,
):
    set_seed(seed, use_cuda=False)

    print(f'Loading data...')
    refine_netlists = [netlist_from_numpy_directory(dataset).original_netlist for dataset in refine_datasets]
    print(f'\t# of samples: {len(refine_netlists)} refine')

    for netlist_name, netlist in zip(refine_datasets, refine_netlists):
        for token in refine_tokens:
            output_file = f'{netlist_name}/output-{token}.npy'
            refine_file = f'{netlist_name}/output-refine-{token}.npy'
            if not os.path.exists(output_file):
                print(f'\t{netlist_name} with {token} output not found.')
                continue
            print(f'\tFor {netlist_name} with {token} output:')
            output_pos = torch.tensor(np.load(output_file), dtype=torch.float32) + netlist.cell_prop_dict['size'] / 2
            layout = layout_from_netlist_cell_pos(netlist, output_pos)
            # refine_pos = refined_layout_pos(layout, use_tqdm=use_tqdm)
            refine_pos = refined_force_pos(layout, use_tqdm=use_tqdm)
            np.save(refine_file,
                    refine_pos - np.array(netlist.cell_prop_dict['size'].cpu().detach(), dtype=np.float32) / 2)

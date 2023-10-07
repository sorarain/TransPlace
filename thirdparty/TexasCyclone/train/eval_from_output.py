import argparse
import numpy as np
import torch
import json
import os
from typing import List, Dict, Any
from time import time
from tqdm import tqdm

from data.graph import Netlist, Layout
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_cell_pos
from data.utils import set_seed
from train.functions import HPWLMetric, RUDYMetric, AreaMetric, OverlapMetric


def eval_from_output(
        name: str,
        args: argparse.Namespace,
        eval_datasets: List[str],
        eval_tokens: List[str],
        log_dir: str = None,
):
    # Configure environment
    log: Dict[str, Any] = {}
    use_cuda = args.device != 'cpu'
    use_tqdm = args.use_tqdm
    device = torch.device(args.device)

    set_seed(args.seed, use_cuda=use_cuda)

    # Load data
    # torch.autograd.set_detect_anomaly(True)
    print(f'Loading data...')
    eval_netlists = [netlist_from_numpy_directory(dataset).original_netlist for dataset in eval_datasets]
    print(f'\t# of samples: {len(eval_netlists)} eval')

    # Calculate metric
    print(f'Calculating metric...')
    hpwl_metric_op = HPWLMetric(device)
    rudy_metric_op = RUDYMetric(use_tqdm=use_tqdm)
    area_metric_op = AreaMetric()
    overlap_metric_op = OverlapMetric()

    def calc_metric(layout: Layout) -> Dict[str, float]:
        print('\t\tcalculating HPWL...')
        hpwl_metric = hpwl_metric_op.calculate(layout)
        print('\t\tcalculating RUDY...')
        rudy_metric = rudy_metric_op.calculate(layout)
        print('\t\tcalculating Area...')
        area_metric = area_metric_op.calculate(layout, limit=[0, 0, *layout.netlist.layout_size])
        print('\t\tcalculating Overlap...')
        overlap_metric = overlap_metric_op.calculate(layout)
        return {
            'hpwl_metric': hpwl_metric,
            'rudy_metric': rudy_metric,
            'area_metric': area_metric,
            'overlap_metric': overlap_metric,
        }

    def evaluate(netlists: List[Netlist], netlist_names: List[str], output_tokens: List[str], verbose=True):
        for netlist_name, netlist in zip(netlist_names, netlists):
            for token in output_tokens:
                output_file = f'{netlist_name}/output-{token}.npy'
                if not os.path.exists(output_file):
                    print(f'\t{netlist_name} with {token} output not found.')
                    continue
                print(f'\tFor {netlist_name} with {token} output:')
                output_pos = torch.tensor(np.load(output_file), dtype=torch.float32) + netlist.cell_prop_dict['size'] / 2
                layout = layout_from_netlist_cell_pos(netlist, output_pos)
                metric_dict = calc_metric(layout)
                hpwl_metric = metric_dict['hpwl_metric']
                rudy_metric = metric_dict['rudy_metric']
                area_metric = metric_dict['area_metric']
                overlap_metric = metric_dict['overlap_metric']
                print(f'\t\tHPWL Metric: {hpwl_metric / 1e9:.2f}B')
                print(f'\t\tRUDY Metric: {rudy_metric / 1e3:.2f}K')
                print(f'\t\tArea Metric: {area_metric / 1e6:.2f}M')
                print(f'\t\tOverlap Metric: {overlap_metric / 1e6:.2f}M')
                d = {
                    f'{netlist_name}-{token}_hpwl_metric': float(hpwl_metric),
                    f'{netlist_name}-{token}_rudy_metric': float(rudy_metric),
                    f'{netlist_name}-{token}_area_metric': float(area_metric),
                    f'{netlist_name}-{token}_overlap_metric': float(overlap_metric),
                }
                log.update(d)
                torch.cuda.empty_cache()

    t2 = time()
    evaluate(eval_netlists, eval_datasets, eval_tokens, verbose=False)

    print("\teval time", time() - t2)
    log.update({'eval_time': time() - t2})
    if log_dir is not None:
        with open(f'{log_dir}/eval-{name}.json', 'w+') as fp:
            json.dump(log, fp)

import argparse
import torch
import json
from typing import List, Dict, Any
from time import time
from tqdm import tqdm

from data.graph import Netlist, Layout
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_ref
from data.utils import set_seed
from train.functions import HPWLMetric, RUDYMetric, AreaMetric, OverlapMetric


def eval_dreamplace(
        name: str,
        args: argparse.Namespace,
        eval_datasets: List[str],
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

    def evaluate(netlists: List[Netlist], netlist_names: List[str], verbose=True):
        for netlist_name, netlist in zip(netlist_names, netlists):
            print(f'\tFor {netlist_name}:')
            layout = layout_from_netlist_ref(netlist)
            metric_dict = calc_metric(layout)
            hpwl_metric = metric_dict['hpwl_metric']
            rudy_metric = metric_dict['rudy_metric']
            area_metric = metric_dict['area_metric']
            overlap_metric = metric_dict['overlap_metric']
            print(f'\t\tHPWL Metric: {hpwl_metric}')
            print(f'\t\tRUDY Metric: {rudy_metric}')
            print(f'\t\tArea Metric: {area_metric}')
            print(f'\t\tOverlap Metric: {overlap_metric}')
            d = {
                f'{netlist_name}_hpwl_metric': float(hpwl_metric),
                f'{netlist_name}_rudy_metric': float(rudy_metric),
                f'{netlist_name}_area_metric': float(area_metric),
                f'{netlist_name}_overlap_metric': float(overlap_metric),
            }
            log.update(d)
            torch.cuda.empty_cache()

    t2 = time()
    evaluate(eval_netlists, eval_datasets, verbose=False)

    print("\teval time", time() - t2)
    log.update({'eval_time': time() - t2})
    if log_dir is not None:
        with open(f'{log_dir}/eval-{name}.json', 'w+') as fp:
            json.dump(log, fp)

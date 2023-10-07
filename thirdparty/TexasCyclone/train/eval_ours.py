import argparse
import torch
import json
import dgl
from typing import List, Dict, Any
from functools import reduce
from copy import copy
from time import time
from tqdm import tqdm

from data.graph import Netlist, Layout, expand_netlist, assemble_layout_with_netlist_info
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_dis_deflect
from data.utils import set_seed
from train.model import NaiveGNN, PlaceGNN
from train.functions import HPWLMetric, RUDYMetric, AreaMetric, OverlapMetric


def eval_ours(
        args: argparse.Namespace,
        eval_datasets: List[str],
        log_dir: str = None,
):
    # Configure environment
    assert args.model, f'Should configure --model'
    log: Dict[str, Any] = {}
    use_cuda = args.device != 'cpu'
    use_tqdm = args.use_tqdm
    device = torch.device(args.device)

    set_seed(args.seed, use_cuda=use_cuda)

    # Load data
    # torch.autograd.set_detect_anomaly(True)
    print(f'Loading data...')
    eval_netlists = [netlist_from_numpy_directory(dataset) for dataset in eval_datasets]
    print(f'\t# of samples: {len(eval_netlists)} eval')

    # Configure model
    print(f'Building model...')
    sample_netlist = eval_netlists[0]
    raw_cell_feats = sample_netlist.cell_prop_dict['feat'].shape[1]
    raw_net_feats = sample_netlist.net_prop_dict['feat'].shape[1]
    raw_pin_feats = sample_netlist.pin_prop_dict['feat'].shape[1]
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'PASS_TYPE': args.pass_type,
    }

    if args.gnn == 'naive':
        model = NaiveGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
    elif args.gnn == 'place':
        config.update({
            'NUM_LAYERS': 1,
            'NUM_HEADS': 2,
        })
        model = PlaceGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
    else:
        assert False, f'Undefined GNN {args.gnn}'

    print(f'\tUsing model model/{args.model}.pkl')
    model_dicts = torch.load(f'model/{args.model}.pkl', map_location=device)
    model.load_state_dict(model_dicts)
    model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

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
            dict_netlist = expand_netlist(netlist)
            iter_i_sub_netlist = tqdm(dict_netlist.items(), total=len(dict_netlist.items()), leave=False) \
                if use_tqdm else dict_netlist.items()
            total_len = len(dict_netlist.items())
            dni: Dict[int, Dict[str, Any]] = {}  # dict_netlist_info

            batch_netlist_id = []
            total_batch_nodes_num = 0
            total_batch_edge_idx = 0
            batch_cell_feature = []
            batch_net_feature = []
            batch_pin_feature = []
            sub_netlist_feature_idrange = []
            batch_cell_size = []
            cnt = 0

            for nid, sub_netlist in iter_i_sub_netlist:
                dni[nid] = {}
                batch_netlist_id.append(nid)
                father, _ = sub_netlist.graph.edges(etype='points-to')
                edge_idx_num = father.size(0)
                sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                total_batch_edge_idx += edge_idx_num
                total_batch_nodes_num += sub_netlist.graph.num_nodes('cell')
                batch_cell_feature.append(sub_netlist.cell_prop_dict['feat'])
                batch_net_feature.append(sub_netlist.net_prop_dict['feat'])
                batch_pin_feature.append(sub_netlist.pin_prop_dict['feat'])
                batch_cell_size.append(sub_netlist.cell_prop_dict['size'])
                if total_batch_nodes_num > 10000 or cnt == total_len - 1:
                    batch_cell_feature = torch.vstack(batch_cell_feature)
                    batch_net_feature = torch.vstack(batch_net_feature)
                    batch_pin_feature = torch.vstack(batch_pin_feature)
                    batch_cell_size = torch.vstack(batch_cell_size)
                    batch_graph = []
                    for nid_ in batch_netlist_id:
                        netlist = dict_netlist[nid_]
                        batch_graph.append(netlist.graph)
                    # batch_graph = dgl.batch([sub_netlist.graph for _,sub_netlist in batch_netlist_id])
                    batch_graph = dgl.batch(batch_graph)
                    batch_edge_dis, batch_edge_angle = model.forward(
                        batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature),batch_cell_size)
                    # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                    for j, nid_ in enumerate(batch_netlist_id):
                        sub_netlist_ = dict_netlist[nid_]
                        begin_idx, end_idx = sub_netlist_feature_idrange[j]
                        edge_dis, edge_angle = \
                            batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                        layout, dis_loss = layout_from_netlist_dis_deflect(sub_netlist_, edge_dis, edge_angle)
                        dni[nid_]['cell_pos'] = copy(layout.cell_pos)
                    batch_netlist_id = []
                    sub_netlist_feature_idrange = []
                    total_batch_nodes_num = 0
                    total_batch_edge_idx = 0
                    batch_cell_feature = []
                    batch_net_feature = []
                    batch_pin_feature = []
                    batch_cell_size = []
                cnt += 1
                torch.cuda.empty_cache()
            layout = assemble_layout_with_netlist_info(dni, dict_netlist, device=device)
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
        with open(f'{log_dir}/eval-{args.name}.json', 'w+') as fp:
            json.dump(log, fp)

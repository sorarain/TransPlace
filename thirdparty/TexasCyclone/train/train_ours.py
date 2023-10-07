import argparse
import numpy as np
import torch
import json
import dgl
from typing import List, Dict, Any
from functools import reduce
from copy import copy
from time import time
from tqdm import tqdm

from data.graph import Netlist, Layout, expand_netlist, sequentialize_netlist, assemble_layout_with_netlist_info
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_dis_deflect
from data.utils import set_seed, mean_dict
from train.model import NaiveGNN, PlaceGNN
from train.functions import AreaLoss, HPWLLoss, SampleOverlapLoss, MacroOverlapLoss, SampleNetOverlapLoss
from train.functions import HPWLMetric, RUDYMetric, AreaMetric, OverlapMetric


def train_ours(
        args: argparse.Namespace,
        train_datasets: List[str],
        valid_datasets: List[str],
        test_datasets: List[str],
        log_dir: str = None,
        fig_dir: str = None,
        model_dir: str = None,
):
    # Configure environment
    logs: List[Dict[str, Any]] = []
    use_cuda = args.device != 'cpu'
    use_tqdm = args.use_tqdm
    device = torch.device(args.device)

    set_seed(args.seed, use_cuda=use_cuda)

    # Load data
    # torch.autograd.set_detect_anomaly(True)
    print(f'Loading data...')
    print(args.use_hierarchical)
    train_netlists = [netlist_from_numpy_directory(dataset,args.use_hierarchical,1) for dataset in train_datasets]
    valid_netlists = [netlist_from_numpy_directory(dataset,args.use_hierarchical,1) for dataset in valid_datasets]
    test_netlists = [netlist_from_numpy_directory(dataset,args.use_hierarchical,1) for dataset in test_datasets]
    print(f'\t# of samples: '
          f'{len(train_netlists)} train, '
          f'{len(valid_netlists)} valid, '
          f'{len(test_netlists)} test.')

    # Configure model
    print(f'Building model...')
    sample_netlist = train_netlists[0] if train_netlists else test_netlists[0]
    raw_cell_feats = sample_netlist.graph.nodes['cell'].data['feat'].shape[1]
    raw_net_feats = sample_netlist.graph.nodes['net'].data['feat'].shape[1]
    raw_pin_feats = sample_netlist.graph.edges['pinned'].data['feat'].shape[1]
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

    if args.model:
        print(f'\tUsing model model/{args.model}.pkl')
        model_dicts = torch.load(f'model/{args.model}.pkl', map_location=device)
        model.load_state_dict(model_dicts)
        model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

    # Train model
    best_metric = 1e8  # lower is better
    evaluate_cell_pos_corner_dict = {}
    # sample_overlap_loss_op = SampleOverlapLoss(span=4)
    macro_overlap_loss_op = MacroOverlapLoss(max_cap=50)
    area_loss_op = AreaLoss()
    hpwl_loss_op = HPWLLoss(device)
    cong_loss_op = SampleNetOverlapLoss(device, span=4)

    hpwl_metric_op = HPWLMetric(device)
    rudy_metric_op = RUDYMetric()
    area_metric_op = AreaMetric()
    overlap_metric_op = OverlapMetric()

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})

        def calc_loss(layout: Layout) -> Dict[str, torch.Tensor]:
            # sample_overlap_loss = sample_overlap_loss_op.forward(layout)
            macro_overlap_loss = macro_overlap_loss_op.forward(layout)
            overlap_loss = macro_overlap_loss * 10
            area_loss = area_loss_op.forward(layout, limit=[0, 0, *layout.netlist.layout_size])
            hpwl_loss = hpwl_loss_op.forward(layout)
            # cong_loss = cong_loss_op.forward(layout)
            # assert not torch.isnan(cong_loss)
            assert not torch.isnan(hpwl_loss)
            assert not torch.isnan(area_loss)
            assert not torch.isnan(macro_overlap_loss)
            # assert not torch.isnan(sample_overlap_loss)
            # assert not torch.isinf(cong_loss)
            assert not torch.isinf(hpwl_loss)
            assert not torch.isinf(area_loss)
            assert not torch.isinf(macro_overlap_loss)
            # assert not torch.isinf(sample_overlap_loss)
            return {
                # 'sample_overlap_loss': sample_overlap_loss,
                'macro_overlap_loss': macro_overlap_loss,
                'overlap_loss': overlap_loss,
                'area_loss': area_loss,
                'hpwl_loss': hpwl_loss,
                # 'cong_loss': cong_loss,
            }

        def calc_metric(layout: Layout) -> Dict[str, float]:
            hpwl_metric = hpwl_metric_op.calculate(layout)
            rudy_metric = rudy_metric_op.calculate(layout)
            area_metric = area_metric_op.calculate(layout, limit=[0, 0, *layout.netlist.layout_size])
            overlap_metric = overlap_metric_op.calculate(layout)
            return {
                'hpwl_metric': hpwl_metric,
                'rudy_metric': rudy_metric,
                'area_metric': area_metric,
                'overlap_metric': overlap_metric,
            }

        def train(netlists: List[Netlist]):
            model.train()
            t1 = time()
            losses = []
            seq_netlists = reduce(lambda x, y: x + y, [sequentialize_netlist(nl) for nl in netlists])
            n_netlist = len(seq_netlists)
            iter_i_netlist = tqdm(enumerate(seq_netlists), total=n_netlist) \
                if use_tqdm else enumerate(seq_netlists)

            batch_netlist = []
            total_batch_nodes_num = 0
            total_batch_edge_idx = 0
            batch_cell_feature = []
            batch_net_feature = []
            batch_pin_feature = []
            sub_netlist_feature_idrange = []
            batch_cell_size = []

            for j, netlist in iter_i_netlist:
                batch_netlist.append(netlist)
                father, _ = netlist.graph.edges(etype='points-to')
                edge_idx_num = father.size(0)
                sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                total_batch_edge_idx += edge_idx_num
                total_batch_nodes_num += netlist.graph.num_nodes('cell')
                batch_cell_feature.append(netlist.graph.nodes['cell'].data['feat'])
                batch_net_feature.append(netlist.graph.nodes['net'].data['feat'])
                batch_pin_feature.append(netlist.graph.edges['pinned'].data['feat'])
                batch_cell_size.append(netlist.graph.nodes['cell'].data['size'])
                if total_batch_nodes_num > 10000 or j == n_netlist - 1:
                    batch_cell_feature = torch.vstack(batch_cell_feature)
                    batch_net_feature = torch.vstack(batch_net_feature)
                    batch_pin_feature = torch.vstack(batch_pin_feature)
                    batch_graph = dgl.batch([sub_netlist.graph for sub_netlist in batch_netlist])
                    batch_cell_size = torch.vstack(batch_cell_size)
                    batch_edge_dis, batch_edge_deflect = model.forward(
                        batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature), batch_cell_size)
                    # batch_edge_dis,batch_edge_deflect = batch_edge_dis.cpu(),batch_edge_deflect.cpu()
                    for nid, sub_netlist in enumerate(batch_netlist):
                        begin_idx, end_idx = sub_netlist_feature_idrange[nid]
                        edge_dis, edge_deflect = batch_edge_dis[begin_idx:end_idx], batch_edge_deflect[begin_idx:end_idx]
                        layout, dis_loss = layout_from_netlist_dis_deflect(sub_netlist, edge_dis, edge_deflect)
                        assert not torch.isnan(dis_loss), f"{dis_loss}"
                        assert not torch.isinf(dis_loss), f"{dis_loss}"
                        loss_dict = calc_loss(layout)
                        loss = sum((
                            args.dis_lambda * dis_loss,
                            args.overlap_lambda * loss_dict['overlap_loss'],
                            args.area_lambda * loss_dict['area_loss'],
                            args.hpwl_lambda * loss_dict['hpwl_loss'],
                            # args.cong_lambda * loss_dict['cong_loss'],
                        ))
                        losses.append(loss)
                    (sum(losses) / len(losses)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                    optimizer.step()
                    losses.clear()
                    batch_netlist = []
                    sub_netlist_feature_idrange = []
                    total_batch_nodes_num = 0
                    total_batch_edge_idx = 0
                    batch_cell_feature = []
                    batch_net_feature = []
                    batch_pin_feature = []
                    batch_cell_size = []
                    torch.cuda.empty_cache()
            print(f"\tTraining time per epoch: {time() - t1}")

        def evaluate(netlists: List[Netlist], dataset_name: str, netlist_names: List[str], verbose=True) -> float:
            model.eval()
            ds = []
            print(f'\tEvaluate {dataset_name}:')
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
                total_dis = []
                total_angle = []
                cnt = 0

                for nid, sub_netlist in iter_i_sub_netlist:
                    dni[nid] = {}
                    batch_netlist_id.append(nid)
                    father, _ = sub_netlist.graph.edges(etype='points-to')
                    edge_idx_num = father.size(0)
                    sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                    total_batch_edge_idx += edge_idx_num
                    total_batch_nodes_num += sub_netlist.graph.num_nodes('cell')
                    batch_cell_feature.append(sub_netlist.graph.nodes['cell'].data['feat'])
                    batch_net_feature.append(sub_netlist.graph.nodes['net'].data['feat'])
                    batch_pin_feature.append(sub_netlist.graph.edges['pinned'].data['feat'])
                    batch_cell_size.append(sub_netlist.graph.nodes['cell'].data['size'])
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
                        batch_edge_dis, batch_edge_deflect = model.forward(
                            batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature),batch_cell_size)
                        # batch_edge_dis,batch_edge_deflect = batch_edge_dis.cpu(),batch_edge_deflect.cpu()
                        total_dis.append(batch_edge_dis.unsqueeze(1))
                        total_angle.append(batch_edge_deflect.unsqueeze(1))
                        for j, nid_ in enumerate(batch_netlist_id):
                            sub_netlist_ = dict_netlist[nid_]
                            begin_idx, end_idx = sub_netlist_feature_idrange[j]
                            edge_dis, edge_deflect = \
                                batch_edge_dis[begin_idx:end_idx], batch_edge_deflect[begin_idx:end_idx]
                            layout, dis_loss = layout_from_netlist_dis_deflect(sub_netlist_, edge_dis, edge_deflect)
                            assert not torch.isnan(dis_loss)
                            assert not torch.isinf(dis_loss)
                            loss_dict = calc_loss(layout)
                            loss_dict = {k: float(v.cpu().clone().detach().data) for k, v in loss_dict.items()}
                            dni[nid_]['dis_loss'] = float(dis_loss.cpu().clone().detach().data)
                            # dni[nid_]['sample_overlap_loss'] = loss_dict['sample_overlap_loss']
                            dni[nid_]['macro_overlap_loss'] = loss_dict['macro_overlap_loss']
                            dni[nid_]['overlap_loss'] = loss_dict['overlap_loss']
                            dni[nid_]['area_loss'] = loss_dict['area_loss']
                            dni[nid_]['hpwl_loss'] = loss_dict['hpwl_loss']
                            # dni[nid_]['cong_loss'] = loss_dict['cong_loss']
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
                # layout = assemble_layout({nid: nif['layout'] for nid, nif in dni.items()}, device=torch.device("cpu"))
                dis_loss = sum(v['dis_loss'] for v in dni.values()) / len(dni)
                # sample_overlap_loss = sum(v['sample_overlap_loss'] for v in dni.values()) / len(dni)
                macro_overlap_loss = sum(v['macro_overlap_loss'] for v in dni.values()) / len(dni)
                overlap_loss = sum(v['overlap_loss'] for v in dni.values()) / len(dni)
                area_loss = sum(v['area_loss'] for v in dni.values()) / len(dni)
                hpwl_loss = sum(v['hpwl_loss'] for v in dni.values()) / len(dni)
                # cong_loss = sum(v['cong_loss'] for v in dni.values()) / len(dni)
                loss = sum((
                    args.dis_lambda * dis_loss,
                    args.overlap_lambda * overlap_loss,
                    args.area_lambda * area_loss,
                    args.hpwl_lambda * hpwl_loss,
                    # args.cong_lambda * cong_loss,
                ))
                metric_dict = calc_metric(layout)
                hpwl_metric = metric_dict['hpwl_metric']
                rudy_metric = metric_dict['rudy_metric']
                area_metric = metric_dict['area_metric']
                overlap_metric = metric_dict['overlap_metric']
                print(f'\t\tDiscrepancy Loss: {dis_loss}')
                # print(f'\t\tSample Overlap Loss: {sample_overlap_loss}')
                print(f'\t\tMacro Overlap Loss: {macro_overlap_loss}')
                print(f'\t\tTotal Overlap Loss: {overlap_loss}')
                print(f'\t\tArea Loss: {area_loss}')
                print(f'\t\tHPWL Loss: {hpwl_loss}')
                # print(f'\t\tCongestion Loss: {cong_loss}')
                print(f'\t\tTotal Loss: {loss}')
                print(f'\t\tHPWL Metric: {hpwl_metric}')
                print(f'\t\tRUDY Metric: {rudy_metric}')
                print(f'\t\tArea Metric: {area_metric}')
                print(f'\t\tOverlap Metric: {overlap_metric}')
                d = {
                    f'{dataset_name}_dis_loss': float(dis_loss),
                    # f'{dataset_name}_sample_overlap_loss': float(sample_overlap_loss),
                    f'{dataset_name}_macro_overlap_loss': float(macro_overlap_loss),
                    f'{dataset_name}_overlap_loss': float(overlap_loss),
                    f'{dataset_name}_area_loss': float(area_loss),
                    f'{dataset_name}_hpwl_loss': float(hpwl_loss),
                    # f'{dataset_name}_cong_loss': float(cong_loss),
                    f'{dataset_name}_loss': float(loss),
                    f'{netlist_name}_hpwl_metric': float(hpwl_metric),
                    f'{netlist_name}_rudy_metric': float(rudy_metric),
                    f'{netlist_name}_area_metric': float(area_metric),
                    f'{netlist_name}_overlap_metric': float(overlap_metric),
                }
                ds.append(d)
                evaluate_cell_pos_corner_dict[netlist_name] = \
                    layout.cell_pos.cpu().detach().numpy() - layout.cell_size.cpu().detach().numpy() / 2
                del loss
                torch.cuda.empty_cache()

            logs[-1].update(mean_dict(ds))
            return logs[-1][f'{dataset_name}_loss']

        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_netlists)
                scheduler.step()
        logs[-1].update({'train_time': time() - t0})
        t2 = time()
        valid_metric = None
        evaluate(train_netlists, 'train', train_datasets, verbose=False)
        if len(valid_netlists):
            valid_metric = evaluate(valid_netlists, 'valid', valid_datasets)
        if len(test_netlists):
            evaluate(test_netlists, 'test', test_datasets)

        if valid_metric is not None and valid_metric < best_metric:
            best_metric = valid_metric
            if model_dir is not None:
                print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')
        for dataset, cell_pos_corner in evaluate_cell_pos_corner_dict.items():
            print(f'\tSaving cell positions to {dataset}/output-{args.name}.npy ...:')
            np.save(f'{dataset}/output-{args.name}.npy', cell_pos_corner)
        evaluate_cell_pos_corner_dict.clear()

        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)

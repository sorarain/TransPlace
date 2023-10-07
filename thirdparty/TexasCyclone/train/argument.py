import argparse


def parse_train_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()

    # Environment settings
    args_parser.add_argument(
        '--no', type=str, default='default',
        help='NAME THIS SCRIPT! The log file, visualized figures and saved model will use this title.'
    )
    args_parser.add_argument(
        '--name', type=str, default='default',
        help='NAME THIS SCRIPT! The log file, visualized figures and saved model will use this title.'
    )
    args_parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed'
    )
    args_parser.add_argument(
        '--device', type=str, default='cpu',
        help='computation device e.g. cpu, cuda:0'
    )
    args_parser.add_argument(
        '--use_tqdm', action='store_true',
        help='if use tqdm to visualize process'
    )

    # Model settings
    args_parser.add_argument(
        '--gnn', type=str, default='naive',
        help='GNN model'
    )
    args_parser.add_argument(
        '--model', type=str, default='',
        help='name of reused model, empty if training a new model'
    )
    args_parser.add_argument(
        '--cell_feats', type=int, default=64,
        help='hidden dim of cell features'
    )
    args_parser.add_argument(
        '--net_feats', type=int, default=64,
        help='hidden dim of net features'
    )
    args_parser.add_argument(
        '--pin_feats', type=int, default=8,
        help='hidden dim of pin features'
    )
    args_parser.add_argument(
        '--pass_type', type=str, default='bidirection',
        help='use single or bidirection message passage'
    )
    args_parser.add_argument(
        '--layers', type=int, default=3,
        help='layer of model'
    )

    # Training settings
    args_parser.add_argument(
        '--use_hierarchical', action='store_true',
        help='if use hierarchical'
    )

    args_parser.add_argument(
        '--lr', type=float, default=1e-5,
        help='learning rate'
    )
    args_parser.add_argument(
        '--lr_decay', type=float, default=5e-2,
        help='learning rate decay'
    )
    args_parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='weight decay'
    )
    args_parser.add_argument(
        '--epochs', type=int, default=20,
        help='epochs'
    )
    args_parser.add_argument(
        '--train_epoch', type=int, default=5,
        help='times of iterating train set per epoch'
    )
    args_parser.add_argument(
        '--batch', type=int, default=10,
        help='# of netlists in a batch'
    )
    args_parser.add_argument(
        '--batch_cells', type=int, default=100000,
        help='# of cells in netlists of a batch'
    )
    args_parser.add_argument(
        '--dis_lambda', type=float, default=1e-1,
        help='weight of discrepancy loss'
    )
    args_parser.add_argument(
        '--overlap_lambda', type=float, default=1e0,
        help='weight of overlap loss'
    )
    args_parser.add_argument(
        '--area_lambda', type=float, default=1e0,
        help='weight of area loss'
    )
    args_parser.add_argument(
        '--hpwl_lambda', type=float, default=1e-2,
        help='weight of HPWL loss'
    )
    args_parser.add_argument(
        '--cong_lambda', type=float, default=1e-3,
        help='weight of congestion loss'
    )

    args_parser.add_argument(
        '--param_json', type=str, default='',
        help='param json path'
    )

    args = args_parser.parse_args()
    return args


def parse_pretrain_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()

    # Environment settings
    args_parser.add_argument(
        '--name', type=str, default='pre-default',
        help='NAME THIS SCRIPT! The log file, visualized figures and saved model will use this title.'
    )
    args_parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed'
    )
    args_parser.add_argument(
        '--device', type=str, default='cpu',
        help='computation device e.g. cpu, cuda:0'
    )
    args_parser.add_argument(
        '--use_tqdm', action='store_true',
        help='if use tqdm to visualize process'
    )

    # Model settings
    args_parser.add_argument(
        '--gnn', type=str, default='naive',
        help='GNN model'
    )
    args_parser.add_argument(
        '--model', type=str, default='',
        help='name of reused model, empty if training a new model'
    )
    args_parser.add_argument(
        '--cell_feats', type=int, default=64,
        help='hidden dim of cell features'
    )
    args_parser.add_argument(
        '--net_feats', type=int, default=64,
        help='hidden dim of net features'
    )
    args_parser.add_argument(
        '--pin_feats', type=int, default=8,
        help='hidden dim of pin features'
    )
    args_parser.add_argument(
        '--pass_type', type=str, default='bidirection',
        help='use single or bidirection message passage'
    )
    args_parser.add_argument(
        '--layers', type=int, default=3,
        help='layer of model'
    )

    # Training settings
    args_parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='learning rate'
    )
    args_parser.add_argument(
        '--lr_decay', type=float, default=5e-2,
        help='learning rate decay'
    )
    args_parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='weight decay'
    )
    args_parser.add_argument(
        '--epochs', type=int, default=20,
        help='epochs'
    )
    args_parser.add_argument(
        '--train_epoch', type=int, default=5,
        help='times of iterating train set per epoch'
    )
    args_parser.add_argument(
        '--batch', type=int, default=10,
        help='# of netlists in a batch'
    )
    args_parser.add_argument(
        '--batch_cells', type=int, default=50000,
        help='# of cells in netlists of a batch'
    )
    args_parser.add_argument(
        '--dis_lambda', type=float, default=1e-3,
        help='weight of distance loss'
    )
    args_parser.add_argument(
        '--angle_lambda', type=float, default=1e-1,
        help='weight of angle loss'
    )

    args_parser.add_argument(
        '--param_json', type=str, default='',
        help='param json path'
    )

    args = args_parser.parse_args()
    return args

from data.utils import check_dir
from train.argument import parse_train_args
from train.eval_dreamplace import eval_dreamplace

LOG_DIR = 'log/dreamplace'
eval_datasets = [
    '../Placement-datasets/dac2012/superblue2',
    '../Placement-datasets/dac2012/superblue3',
    '../Placement-datasets/dac2012/superblue6',
    '../Placement-datasets/dac2012/superblue7',
    '../Placement-datasets/dac2012/superblue9',
]

if __name__ == '__main__':
    check_dir(LOG_DIR)
    args = parse_train_args()
    eval_dreamplace(
        name='default',
        args=args,
        eval_datasets=eval_datasets,
        log_dir=LOG_DIR,
    )

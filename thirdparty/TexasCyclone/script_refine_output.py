from data.utils import check_dir
from train.argument import parse_train_args
from train.refine_output import refine_output

LOG_DIR = 'log/eval'
refine_datasets = [
    '../Placement-datasets/dac2012/superblue2',
    # '../Placement-datasets/dac2012/superblue3',
    # '../Placement-datasets/dac2012/superblue6',
    # '../Placement-datasets/dac2012/superblue7',
    # '../Placement-datasets/dac2012/superblue9',
]
refine_tokens = [
    'xpre',
]

if __name__ == '__main__':
    check_dir(LOG_DIR)
    args = parse_train_args()
    refine_output(
        refine_datasets=refine_datasets,
        refine_tokens=refine_tokens,
        seed=0, use_tqdm=True
    )

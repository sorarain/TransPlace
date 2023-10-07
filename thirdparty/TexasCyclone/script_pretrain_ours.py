from data.utils import check_dir
from train.argument import parse_pretrain_args
from train.pretrain_ours import pretrain_ours

LOG_DIR = 'log/pretrain'
FIG_DIR = 'visualize/pretrain'
MODEL_DIR = 'model'
train_datasets = [
    '../Placement-datasets/dac2012/superblue2',
    '../Placement-datasets/dac2012/superblue3',
    '../Placement-datasets/dac2012/superblue6',
    # 'data/test/dataset1/large',
]
valid_datasets = [
    '../Placement-datasets/dac2012/superblue7',
    # 'data/test/dataset1/large-noclu',
]
test_datasets = [
    '../Placement-datasets/dac2012/superblue9',
    # 'data/test/dataset1/small',
]

if __name__ == '__main__':
    check_dir(LOG_DIR)
    check_dir(FIG_DIR)
    check_dir(MODEL_DIR)
    args = parse_pretrain_args()
    pretrain_ours(
        args=args,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        test_datasets=test_datasets,
        log_dir=LOG_DIR,
        fig_dir=FIG_DIR,
        model_dir=MODEL_DIR
    )

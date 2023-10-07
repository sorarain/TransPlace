from data.utils import check_dir
from train.argument import parse_train_args
from train.train_ours import train_ours

LOG_DIR = 'log/ours'
FIG_DIR = 'visualize/ours'
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
    args = parse_train_args()
    train_ours(
        args=args,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        test_datasets=test_datasets,
        log_dir=LOG_DIR,
        fig_dir=FIG_DIR,
        model_dir=MODEL_DIR
    )

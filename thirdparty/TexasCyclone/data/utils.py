import torch
import numpy as np
import tqdm
import os
from typing import List, Dict


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def check_dir(directory: str):
    if not os.path.isdir(directory):
        os.system(f"mkdir -p {directory}")


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    full_dict: Dict[str, List[float]] = {}
    for d in dicts:
        for k, v in d.items():
            full_dict.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in full_dict.items()}

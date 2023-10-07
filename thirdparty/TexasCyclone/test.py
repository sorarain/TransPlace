import numpy as np
import torch


if __name__ == '__main__':
    a = torch.tensor([1, 1, 2, 3])
    b = torch.tensor([0, 1, 3, 0])
    print(torch.arctan2(a, b))
    c = torch.vstack([a, b])
    print(torch.min(c, dim=1)[0])

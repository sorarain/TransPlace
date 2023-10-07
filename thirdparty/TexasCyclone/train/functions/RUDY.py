import numpy as np
from tqdm import tqdm

from data.graph import Layout
from .function import MetricFunction


class RUDYMetric(MetricFunction):
    def __init__(self, w=800, h=800, use_tqdm=False):
        super(RUDYMetric, self).__init__()
        self.w, self.h = w, h
        self.use_tqdm = use_tqdm

    def calculate(self, layout: Layout, *args, **kwargs) -> float:
        net_span = np.array(layout.net_span.cpu().clone().detach(), dtype=np.float32)
        net_degree = np.array(layout.netlist.graph.nodes['net'].data['degree'], dtype=np.float32)
        layout_size = layout.netlist.layout_size
        shape = (int(layout_size[0] / self.w) + 1, int(layout_size[1] / self.h) + 1)
        cong_map = np.zeros(shape=shape, dtype=np.float32)

        iter_net_span_degree = tqdm(zip(net_span, net_degree), total=net_span.shape[0]) \
            if self.use_tqdm else zip(net_span, net_degree)
        for span, (degree,) in iter_net_span_degree:
            w1, w2 = map(int, span[[0, 2]] / self.w)
            h1, h2 = map(int, span[[1, 3]] / self.h)
            density = degree / (w2 - w1 + 1) / (h2 - h1 + 1)
            w2 = min(w2 + 1, shape[0])
            h2 = min(h2 + 1, shape[1])
            if w2 <= w1 or h2 <= h1:
                continue
            cong_map[w1: w2, h1: h2] += density

        return float(np.max(cong_map))

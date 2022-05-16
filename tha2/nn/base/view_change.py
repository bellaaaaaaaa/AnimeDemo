from typing import List

from torch import Tensor
from torch.nn import Module


class ViewChange(Module):
    def __init__(self, new_size: List[int]):
        super().__init__()
        self.new_size = new_size

    def forward(self, x: Tensor):
        n = x.shape[0]
        return x.view([n] + self.new_size)


class ViewImageAsVector(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        try:
            assert x.dim() == 4, "Dimension of x must be 4"
        except AssertionError as msg:
            print(msg)
        n, c, w, h = x.shape
        return x.view(n, c * w * h)


class ViewVectorAsMultiChannelImage(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        try:
            assert x.dim() == 2, "Dimension of x must be 2"
        except AssertionError as msg:
            print(msg)
        n, c = x.shape
        return x.view(n, c, 1, 1)


class ViewVectorAsOneChannelImage(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        try:
            assert x.dim() == 2, "Dimension of x must be 2"
        except AssertionError as msg:
            print(msg)
        n, c = x.shape
        return x.view(n, 1, c, 1)

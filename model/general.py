import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, shape: tuple, act, act_out):
        super(MLP, self).__init__()
        s = []
        for i, j in zip(shape[:-1], shape[1:]):
            s.append(nn.Linear(i, j))
            s.append(act())
        s[-1] = act_out()
        self.seq = nn.Sequential(*s)

    def forward(self, feature):
        return self.seq(feature)
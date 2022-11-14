import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class QLearning(nn):
    def __init__(self):
        super(QLearning, self).__init__()
        self.preprocess = transforms.Compose([
            transforms.resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.2, .2, .2])
        ])
        self.resnet = models.resnet18(pretrained=True)

    def forward(self, state, preprocessed=False):
        s = state if preprocessed else self.preprocess(state)
        a = self.resnet(s)
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef, rpc_sync

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)

class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
    def forward(self, x):
        return self.seq(x)

class Backend(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.rest = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.rest(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class DistResNet(nn.Module):
    def __init__(self, backend_rref):
        super().__init__()
        self.frontend = Frontend()
        self.backend_rref = backend_rref

    def forward(self, x):
        x = self.frontend(x)
        return rpc_sync(self.backend_rref.owner(), _call_backend_forward, args=(self.backend_rref, x))

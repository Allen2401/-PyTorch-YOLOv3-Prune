import torch.nn as nn
import torch
from models.channel_selection import channel_selection

class Bottleneck(nn.Module):

    def __init__(self, in_places, places,cfg, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_places),
            channel_selection(in_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cfg[0],out_channels=cfg[1], kernel_size=1,bias=False),
            nn.BatchNorm2d(cfg[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cfg[1],out_channels=cfg[2],kernel_size=3,stride = stride,padding=1,bias=False),
            nn.BatchNorm2d(cfg[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cfg[2],out_channels=places * expansion,kernel_size=1,stride=1,bias = False),
        )
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
            )

    def forward(self, x):

        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual

        return out

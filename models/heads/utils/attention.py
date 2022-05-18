# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule

class SE(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class Spatial(nn.Module):

    def __init__(self,
                 ratio,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)

        self.conv1 = ConvModule(
            in_channels=256,
            out_channels=1,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1, 
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x * out

class Attention(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        
        self.se = SE(channels, ratio, conv_cfg, act_cfg)

        self.sse = Spatial(ratio, conv_cfg, act_cfg)

    def forward(self, x):
        out1 = self.se(x)
        out2 = self.sse(x)
        return x + out1 + out2
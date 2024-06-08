# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule

from opencd.models.utils.builder import ITERACTION_LAYERS

@ITERACTION_LAYERS.register_module()
class CEFI(BaseModule):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.depth_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                    groups=channel)
        self.point_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                    groups=1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y1_avg = self.avg_pool(x1).view(b, c)
        y1_max = self.max_pool(x1).view(b, c)
        y1 = y1_avg + y1_max
        y1 = self.fc(y1).view(b, c, 1, 1)

        gsconv_x1 = self.depth_conv(x1)
        gsconv_out1 = self.point_conv(gsconv_x1)
        gsconv_out1 = gsconv_out1 + x1

        gsconv_x2 = self.depth_conv(x2)
        gsconv_out2 = self.point_conv(gsconv_x2)
        gsconv_out2 = gsconv_out2 + x2
        out11 = x1 * y1.expand_as(x1)
        out12 = gsconv_out2 * y1.expand_as(x1)
        out_last1 = out11 + out12 + gsconv_out1

        y2_avg = self.avg_pool(x2).view(b, c)
        y2_max = self.max_pool(x2).view(b, c)
        y2 = y2_avg + y2_max
        y2 = self.fc(y2).view(b, c, 1, 1)
        out21 = x2 * y2.expand_as(x2)
        out22 = gsconv_out1 * y2.expand_as(x2)
        out_last2 = out21 + out22 + gsconv_out2

        return out_last1, out_last2

@ITERACTION_LAYERS.register_module()
class SEFI(BaseModule):
    def __init__(self,in_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # map尺寸不变，缩减通道
        avgout1 = torch.mean(x1, dim=1, keepdim=True)
        avgout2 = torch.mean(x2, dim=1, keepdim=True)
        maxout1, _ = torch.max(x1, dim=1, keepdim=True)
        maxout2, _ = torch.max(x2, dim=1, keepdim=True)
        out1 = torch.cat([avgout1, maxout2], dim=1)
        out2 = torch.cat([avgout2, maxout1], dim=1)
        out1 = self.sigmoid(self.conv2d(out1))
        out2 = self.sigmoid(self.conv2d(out2))
        out_1 = out1 * x1
        out_2 = out2 * x2

        return out_1, out_2
#########################################################################################################################
@ITERACTION_LAYERS.register_module()
class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2
#########################################################################################################################
@ITERACTION_LAYERS.register_module()
class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self, 
                 channels, 
                 num_paths=2, 
                 attn_channels=None, 
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


@ITERACTION_LAYERS.register_module()
class TwoIdentity(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        return x1, x2

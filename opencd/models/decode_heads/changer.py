# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from torch.nn import functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS
from ..necks.feature_fusion import FeatureFusionNeck

#from torchvision.ops import deform_conv2d
###################################################################################################
class AdaptiveConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, shrinkage_rate=0.25):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(AdaptiveConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        hidden_dim = int(inc * shrinkage_rate)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, outc, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        u = self.gate(x)
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        out = u * out
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
##################################################################################################
# class SPConv_3x3(nn.Module):
#     def __init__(self, inplanes, outplanes, stride=1, ratio=0.5):
#         super(SPConv_3x3, self).__init__()
#         self.inplanes_3x3 = int(inplanes*ratio)
#         self.inplanes_1x1 = inplanes - self.inplanes_3x3
#         self.outplanes_3x3 = int(outplanes*ratio)
#         self.outplanes_1x1 = outplanes - self.outplanes_3x3
#         self.outplanes = outplanes
#         self.stride = stride

#         self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
#                              padding=1, groups=2, bias=False)
#         self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)
#         self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes,kernel_size=1)
#         self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2,stride=2)
#         self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
#         self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
#         self.bn1 = nn.BatchNorm2d(self.outplanes)
#         self.bn2 = nn.BatchNorm2d(self.outplanes)
#         self.ratio = ratio
#         self.groups = int(1/self.ratio)
#     def forward(self, x):
#         b, c, _, _ = x.size()

#         x_3x3 = x[:,:int(c*self.ratio),:,:]
#         x_1x1 = x[:,int(c*self.ratio):,:,:]
#         out_3x3_gwc = self.gwc(x_3x3)
#         if self.stride ==2:
#             x_3x3 = self.avgpool_s2_3(x_3x3)
#         out_3x3_pwc = self.pwc(x_3x3)
#         out_3x3 = out_3x3_gwc + out_3x3_pwc
#         out_3x3 = self.bn1(out_3x3)
#         out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze(dim=3).squeeze(dim=2)

#         # use avgpool first to reduce information lost
#         if self.stride == 2:
#             x_1x1 = self.avgpool_s2_1(x_1x1)

#         out_1x1 = self.conv1x1(x_1x1)
#         out_1x1 = self.bn2(out_1x1)
#         out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze(dim=3).squeeze(dim=2)

#         out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
#         out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
#         out = out_1x1 * (out_31_ratio[:,:,1].view(b, self.outplanes, 1, 1).expand_as(out_1x1))\
#               + out_3x3 * (out_31_ratio[:,:,0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

#         return out
###################################################################################################
# class MSPoolAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         pools = [3,7,11]
#         self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
#         self.pool1_1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0]//2, count_include_pad=False)
#         self.pool1_2 = nn.MaxPool2d(pools[0], stride=1, padding=pools[0]//2)
#         self.pool2_1 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1]//2, count_include_pad=False)
#         self.pool2_2 = nn.MaxPool2d(pools[1], stride=1, padding=pools[1]//2)
#         self.pool3_1 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2]//2, count_include_pad=False)
#         self.pool3_2 = nn.MaxPool2d(pools[2], stride=1, padding=pools[2]//2)
#         self.conv4 = nn.Conv2d(dim, dim, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.spconv = SPConv_3x3(dim,dim)
        
#     def forward(self, x):
#         #u = x.clone()
#         u =  self.spconv(x)
#         x_in = self.conv0(x)
#         x1_1 = self.pool1_1(x_in)
#         x1_2 = self.pool1_2(x_in)
#         x_1 = x1_1 + x1_2
#         x2_1 = self.pool2_1(x_in)
#         x2_2 = self.pool2_2(x_in)
#         x_2 = x2_1 + x2_2
#         x3_1 = self.pool3_1(x_in)
#         x3_2 = self.pool3_2(x_in)        
#         x_3 = x3_1 + x3_2
#         x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
#         return x_out + u
#         return u
###################################################################################################
# class SqueezeExcitation(nn.Module):
#     def __init__(self, dim, shrinkage_rate=0.25):
#         super().__init__()
#         hidden_dim = int(dim * shrinkage_rate)
#         self.gate0 = nn.AdaptiveMaxPool2d(1)
#         self.gate1 = nn.AdaptiveAvgPool2d(1)
#         self.gate2 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
#         self.gelu = nn.GELU()
#         self.gate3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
#         self.gate4 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         u1 = self.gelu(self.gate2(self.gate0(x)+self.gate1(x)))
#         u2 = self.sigmoid(self.gelu(self.gate4(self.gelu(self.gate3(u1)))))
#         out = x*u2
#         return out

# class MBConv(nn.Module):
#     def __init__(self, dim, growth_rate=2.0):
#         super().__init__()
#         hidden_dim = int(dim * growth_rate)

#         self.mbconv = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 1, 1, 0),
#             nn.GELU(),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
#             nn.GELU(),
#             SqueezeExcitation(hidden_dim),
#             nn.Conv2d(hidden_dim, 4, 1, 1, 0)
#         )

#     def forward(self, x):
#         return self.mbconv(x)
####################################################################################################
class FDAF(BaseModule):
    """Flow Dual-Alignment Fusion Module.
    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg=None
        norm_cfg=dict(type='IN')
        act_cfg=dict(type='GELU')
        
        kernel_size = 5
        self.flow_make = Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            
        )
        #self.MSPoolAttention = MBConv(in_channels*2)
        self.flow_makelast = nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False)
        self.AdaptiveDeformableConvolution = AdaptiveConv2d(in_channels, in_channels)
        
    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""
        u1 = self.AdaptiveDeformableConvolution(x1)
        u2 = self.AdaptiveDeformableConvolution(x2)
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        #flow = self.MSPoolAttention(flow)
        flow = self.flow_makelast(flow)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(u1, f1) - u2
        x2_feat = self.warp(u2, f2) - u1
        
        if fusion_policy == None:
            return x1_feat, x2_feat
        
        output = FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output

class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head of Changer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

@MODELS.register_module()
class Changer(BaseDecodeHead):
    """The Head of Changer.
    This head is the implementation of
    `Changer <https://arxiv.org/abs/2209.08290>` _.
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.neck_layer = FDAF(in_channels=self.channels // 2)
        
        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
                
    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = self.neck_layer(out1, out2, 'concat')

        out = self.discriminator(out)
        out = self.cls_seg(out)

        return out
###############################################################################################
# # Copyright (c) Open-CD. All rights reserved.
# import torch
# import torch.nn as nn
# from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
# from mmcv.cnn.bricks.drop import build_dropout
# from mmengine.model import BaseModule, Sequential
# from torch.nn import functional as F

# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.models.utils import resize
# from opencd.registry import MODELS
# from ..necks.feature_fusion import FeatureFusionNeck


# class FDAF(BaseModule):
#     """Flow Dual-Alignment Fusion Module.
#     Args:
#         in_channels (int): Input channels of features.
#         conv_cfg (dict | None): Config of conv layers.
#             Default: None
#         norm_cfg (dict | None): Config of norm layers.
#             Default: dict(type='BN')
#         act_cfg (dict): Config of activation layers.
#             Default: dict(type='ReLU')
#     """

#     def __init__(self,
#                  in_channels,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='IN'),
#                  act_cfg=dict(type='GELU')):
#         super(FDAF, self).__init__()
#         self.in_channels = in_channels
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         # TODO
#         conv_cfg=None
#         norm_cfg=dict(type='IN')
#         act_cfg=dict(type='GELU')
        
#         kernel_size = 5
#         self.flow_make = Sequential(
#             nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
#             nn.InstanceNorm2d(in_channels*2),
#             nn.GELU(),
#             nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False),
#         )

#     def forward(self, x1, x2, fusion_policy=None):
#         """Forward function."""

#         output = torch.cat([x1, x2], dim=1)
#         flow = self.flow_make(output)
#         f1, f2 = torch.chunk(flow, 2, dim=1)
#         x1_feat = self.warp(x1, f1) - x2
#         x2_feat = self.warp(x2, f2) - x1
        
#         if fusion_policy == None:
#             return x1_feat, x2_feat
        
#         output = FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)
#         return output

#     @staticmethod
#     def warp(x, flow):
#         n, c, h, w = x.size()

#         norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
#         col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
#         row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
#         grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
#         grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
#         grid = grid + flow.permute(0, 2, 3, 1) / norm

#         output = F.grid_sample(x, grid, align_corners=True)
#         return output
# ####################################################################################################
# class DeformConv2d(nn.Module):
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
#         """
#         Args:
#             modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
#         """
#         super(DeformConv2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.zero_padding = nn.ZeroPad2d(padding)
#         self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

#         self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#         nn.init.constant_(self.p_conv.weight, 0)
#         self.p_conv.register_backward_hook(self._set_lr)

#         self.modulation = modulation
#         if modulation:
#             self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#             nn.init.constant_(self.m_conv.weight, 0)
#             self.m_conv.register_backward_hook(self._set_lr)

#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
#         grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

#     def forward(self, x):
#         offset = self.p_conv(x)
#         if self.modulation:
#             m = torch.sigmoid(self.m_conv(x))

#         dtype = offset.data.type()
#         ks = self.kernel_size
#         N = offset.size(1) // 2

#         if self.padding:
#             x = self.zero_padding(x)

#         # (b, 2N, h, w)
#         p = self._get_p(offset, dtype)

#         # (b, h, w, 2N)
#         p = p.contiguous().permute(0, 2, 3, 1)
#         q_lt = p.detach().floor()
#         q_rb = q_lt + 1

#         q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

#         # clip p
#         p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

#         # bilinear kernel (b, h, w, N)
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

#         # (b, c, h, w, N)
#         x_q_lt = self._get_x_q(x, q_lt, N)
#         x_q_rb = self._get_x_q(x, q_rb, N)
#         x_q_lb = self._get_x_q(x, q_lb, N)
#         x_q_rt = self._get_x_q(x, q_rt, N)

#         # (b, c, h, w, N)
#         x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
#                    g_rb.unsqueeze(dim=1) * x_q_rb + \
#                    g_lb.unsqueeze(dim=1) * x_q_lb + \
#                    g_rt.unsqueeze(dim=1) * x_q_rt

#         # modulation
#         if self.modulation:
#             m = m.contiguous().permute(0, 2, 3, 1)
#             m = m.unsqueeze(dim=1)
#             m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
#             x_offset *= m

#         x_offset = self._reshape_x_offset(x_offset, ks)
#         out = self.conv(x_offset)

#         return out

#     def _get_p_n(self, N, dtype):
#         p_n_x, p_n_y = torch.meshgrid(
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
#         # (2N, 1)
#         p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
#         p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

#         return p_n

#     def _get_p_0(self, h, w, N, dtype):
#         p_0_x, p_0_y = torch.meshgrid(
#             torch.arange(1, h*self.stride+1, self.stride),
#             torch.arange(1, w*self.stride+1, self.stride))
#         p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

#         return p_0

#     def _get_p(self, offset, dtype):
#         N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

#         # (1, 2N, 1, 1)
#         p_n = self._get_p_n(N, dtype)
#         # (1, 2N, h, w)
#         p_0 = self._get_p_0(h, w, N, dtype)
#         p = p_0 + p_n + offset
#         return p

#     def _get_x_q(self, x, q, N):
#         b, h, w, _ = q.size()
#         padded_w = x.size(3)
#         c = x.size(1)
#         # (b, c, h*w)
#         x = x.contiguous().view(b, c, -1)

#         # (b, h, w, N)
#         index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
#         # (b, c, h*w*N)
#         index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

#         x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

#         return x_offset

#     @staticmethod
#     def _reshape_x_offset(x_offset, ks):
#         b, c, h, w, N = x_offset.size()
#         x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
#         x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

#         return x_offset

# class attention2d(nn.Module):
#     def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
#         super(attention2d, self).__init__()
#         assert temperature%3==1
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         if in_planes!=3:
#             hidden_planes = int(in_planes*ratios)+1
#         else:
#             hidden_planes = K
#         self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
#         # self.bn = nn.BatchNorm2d(hidden_planes)
#         self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
#         self.temperature = temperature
#         if init_weight:
#             self._initialize_weights()


#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m ,nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def updata_temperature(self):
#         if self.temperature!=1:
#             self.temperature -=3
#             print('Change temperature to:', str(self.temperature))


#     def forward(self, x):
#         x_1 = self.avgpool(x)
#         x_2 = self.maxpool(x)
#         x_temp = x_1 + x_2 
#         x_last = self.fc1(x_temp)
#         x_last = F.relu(x_last)
#         x_last = self.fc2(x_last).view(x_last.size(0), -1)
#         return F.softmax(x_last/self.temperature, 1)


# class Dynamic_conv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=1, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
#         super(Dynamic_conv2d, self).__init__()
#         assert in_planes%groups==0
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.K = K
#         self.attention = attention2d(in_planes, ratio, K, temperature)

#         self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(K, out_planes))
#         else:
#             self.bias = None
#         if init_weight:
#             self._initialize_weights()
#         self.deformconv = DeformConv2d(out_planes, out_planes)

#         #TODO 初始化
#     def _initialize_weights(self):
#         for i in range(self.K):
#             nn.init.kaiming_uniform_(self.weight[i])


#     def update_temperature(self):
#         self.attention.updata_temperature()

#     def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
#         softmax_attention = self.attention(x)
#         batch_size, in_planes, height, width = x.size()
#         x = x.contiguous().view(1, -1, height, width)# 变化成一个维度进行组卷积
#         weight = self.weight.view(self.K, -1)

#         # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
#         aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
#         if self.bias is not None:
#             aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
#             output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups*batch_size)
#         else:
#             output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups * batch_size)

#         output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
#         output = self.deformconv(output)
#         return output
# ####################################################################################################
# class SPConv_3x3(nn.Module):
#     def __init__(self, inplanes, outplanes, stride=1, ratio=0.5):
#         super(SPConv_3x3, self).__init__()
#         self.inplanes_3x3 = int(inplanes*ratio)
#         self.inplanes_1x1 = inplanes - self.inplanes_3x3
#         self.outplanes_3x3 = int(outplanes*ratio)
#         self.outplanes_1x1 = outplanes - self.outplanes_3x3
#         self.outplanes = outplanes
#         self.stride = stride

#         self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
#                              padding=1, groups=2, bias=False)
#         self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)

#         #self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes,kernel_size=1)
#         self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2,stride=2)
#         self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
#         self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
#         self.bn1 = nn.BatchNorm2d(self.outplanes)
#         self.bn2 = nn.BatchNorm2d(self.outplanes)
#         self.ratio = ratio
#         self.groups = int(1/self.ratio)
#         self.conv1x1 = Dynamic_conv2d(in_planes= self.inplanes_1x1, out_planes= self.outplanes, kernel_size=3)
#     def forward(self, x):
#         b, c, _, _ = x.size()


#         x_3x3 = x[:,:int(c*self.ratio),:,:]
#         x_1x1 = x[:,int(c*self.ratio):,:,:]
#         out_3x3_gwc = self.gwc(x_3x3)
#         if self.stride ==2:
#             x_3x3 = self.avgpool_s2_3(x_3x3)
#         out_3x3_pwc = self.pwc(x_3x3)
#         out_3x3 = out_3x3_gwc + out_3x3_pwc
#         out_3x3 = self.bn1(out_3x3)
#         out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze(dim=3).squeeze(dim=2)
        
#         # use avgpool first to reduce information lost
#         if self.stride == 2:
#             x_1x1 = self.avgpool_s2_1(x_1x1)

#         out_1x1 = self.conv1x1(x_1x1)
        
#         out_1x1 = self.bn2(out_1x1)
        
#         out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze(dim=3).squeeze(dim=2)

#         out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
#         out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
#         out = out_1x1 * (out_31_ratio[:,:,1].view(b, self.outplanes, 1, 1).expand_as(out_1x1))\
#               + out_3x3 * (out_31_ratio[:,:,0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

#         return out
# ####################################################################################################
# class MSPoolAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         pools = [3,7,11]
#         self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
#         self.pool1_1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0]//2, count_include_pad=False)
#         self.pool1_2 = nn.MaxPool2d(pools[0], stride=1, padding=pools[0]//2)
#         self.pool2_1 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1]//2, count_include_pad=False)
#         self.pool2_2 = nn.MaxPool2d(pools[1], stride=1, padding=pools[1]//2)
#         self.pool3_1 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2]//2, count_include_pad=False)
#         self.pool3_2 = nn.MaxPool2d(pools[2], stride=1, padding=pools[2]//2)
#         self.conv4 = nn.Conv2d(dim, dim, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.spconv = SPConv_3x3(dim,dim)
        
#     def forward(self, x):
#         #u = x.clone()
#         u =  self.spconv(x)
#         x_in = self.conv0(x)
#         x1_1 = self.pool1_1(x_in)
#         x1_2 = self.pool1_2(x_in)
#         x_1 = x1_1 + x1_2
#         x2_1 = self.pool2_1(x_in)
#         x2_2 = self.pool2_2(x_in)
#         x_2 = x2_1 + x2_2
#         x3_1 = self.pool3_1(x_in)
#         x3_2 = self.pool3_2(x_in)        
#         x_3 = x3_1 + x3_2
#         x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
#         return x_out + u
# ####################################################################################################
# class MixFFN(BaseModule):
#     """An implementation of MixFFN of Segformer. \
#         Here MixFFN is uesd as projection head of Changer.
#     Args:
#         embed_dims (int): The feature dimension. Same as
#             `MultiheadAttention`. Defaults: 256.
#         feedforward_channels (int): The hidden dimension of FFNs.
#             Defaults: 1024.
#         act_cfg (dict, optional): The activation config for FFNs.
#             Default: dict(type='ReLU')
#         ffn_drop (float, optional): Probability of an element to be
#             zeroed in FFN. Default 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#     """

#     def __init__(self,
#                  embed_dims,
#                  feedforward_channels,
#                  act_cfg=dict(type='GELU'),
#                  ffn_drop=0.,
#                  dropout_layer=None,
#                  init_cfg=None):
#         super(MixFFN, self).__init__(init_cfg)

#         self.embed_dims = embed_dims
#         self.feedforward_channels = feedforward_channels
#         self.act_cfg = act_cfg
#         self.activate = build_activation_layer(act_cfg)
#         in_channels = embed_dims
#         self.fc1 = Conv2d(
#             in_channels=in_channels,
#             out_channels=feedforward_channels*2,
#             kernel_size=1,
#             stride=1,
#             bias=True)
#         self.gelu = nn.GELU()
#         self.pe_conv1_1 = Conv2d(
#             in_channels=feedforward_channels*2,
#             out_channels=feedforward_channels*2,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=True,
#             groups=feedforward_channels*2)
#         self.pe_conv1_2 = Conv2d(
#             in_channels=feedforward_channels*2,
#             out_channels=feedforward_channels*2,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             bias=True,
#             groups=feedforward_channels*2)
#         self.pe_conv2_1 = Conv2d(
#             in_channels=feedforward_channels*2,
#             out_channels=feedforward_channels,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             bias=True,
#             groups=feedforward_channels)
#         self.pe_conv2_2 = Conv2d(
#             in_channels=feedforward_channels*2,
#             out_channels=feedforward_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=True,
#             groups=feedforward_channels)
#         self.fc2 = Conv2d(
#             in_channels=feedforward_channels*2,
#             out_channels=in_channels,
#             kernel_size=1,
#             stride=1,
#             bias=True)
#         self.drop = nn.Dropout(ffn_drop)
#         #layers = [fc1, pe_conv1, pe_conv2, self.activate, drop, fc2, drop]
#         #self.layers = Sequential(*layers)
#         self.dropout_layer = build_dropout(
#             dropout_layer) if dropout_layer else torch.nn.Identity()
#         self.MSPoolAttention = MSPoolAttention(in_channels)
#     def forward(self, x, identity=None):
#         out1 = self.fc1(x)
#         out1_11, out1_12 = self.gelu(self.pe_conv1_1(out1)).chunk(2, dim=1) 
#         out1_21, out1_22 = self.gelu(self.pe_conv1_2(out1)).chunk(2, dim=1)
#         out1_1 = torch.cat([out1_11, out1_21], dim=1)
#         out1_2 = torch.cat([out1_12, out1_22], dim=1)
        
#         out2_1 = self.gelu(self.pe_conv2_1(out1_1))
#         out2_2 = self.gelu(self.pe_conv2_2(out1_2))
#         out2 = torch.cat([out2_1, out2_2], dim=1)
#         out = self.activate(out2)
#         out = self.drop(out)
#         out = self.fc2(out)
#         out = self.drop(out)

#         if identity is None:
#             identity = self.MSPoolAttention(x)
#         return identity + self.dropout_layer(out)
# # ####################################################################################################
# @MODELS.register_module()
# class Changer(BaseDecodeHead):
#     """The Head of Changer.
#     This head is the implementation of
#     `Changer <https://arxiv.org/abs/2209.08290>` _.
#     Args:
#         interpolate_mode: The interpolate mode of MLP head upsample operation.
#             Default: 'bilinear'.
#     """

#     def __init__(self, interpolate_mode='bilinear', **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)

#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)
#         assert num_inputs == len(self.in_index)

#         self.convs = nn.ModuleList()
#         for i in range(num_inputs):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))

#         self.fusion_conv = ConvModule(
#             in_channels=self.channels * num_inputs,
#             out_channels=self.channels // 2,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)
        
#         self.neck_layer = FDAF(in_channels=self.channels // 2)
        
#         # projection head
#         self.discriminator = MixFFN(
#             embed_dims=self.channels,
#             feedforward_channels=self.channels,
#             ffn_drop=0.,
#             dropout_layer=dict(type='DropPath', drop_prob=0.),
#             act_cfg=dict(type='GELU'))
                
#     def base_forward(self, inputs):
#         outs = []
#         for idx in range(len(inputs)):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             outs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners))

#         out = self.fusion_conv(torch.cat(outs, dim=1))
        
#         return out

#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         inputs1 = []
#         inputs2 = []
#         for input in inputs:
#             f1, f2 = torch.chunk(input, 2, dim=1)
#             inputs1.append(f1)
#             inputs2.append(f2)
        
#         out1 = self.base_forward(inputs1)
#         out2 = self.base_forward(inputs2)
#         out = self.neck_layer(out1, out2, 'concat')

#         out = self.discriminator(out)
#         out = self.cls_seg(out)

#         return out
############################################################################################
# # Copyright (c) Open-CD. All rights reserved.
# import torch
# import torch.nn as nn
# from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
# from mmcv.cnn.bricks.drop import build_dropout
# from mmengine.model import BaseModule, Sequential
# from torch.nn import functional as F

# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.models.utils import resize
# from opencd.registry import MODELS
# from ..necks.feature_fusion import FeatureFusionNeck


# class FDAF(BaseModule):
#     """Flow Dual-Alignment Fusion Module.

#     Args:
#         in_channels (int): Input channels of features.
#         conv_cfg (dict | None): Config of conv layers.
#             Default: None
#         norm_cfg (dict | None): Config of norm layers.
#             Default: dict(type='BN')
#         act_cfg (dict): Config of activation layers.
#             Default: dict(type='ReLU')
#     """

#     def __init__(self,
#                  in_channels,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='IN'),
#                  act_cfg=dict(type='GELU')):
#         super(FDAF, self).__init__()
#         self.in_channels = in_channels
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         # TODO
#         conv_cfg=None
#         norm_cfg=dict(type='IN')
#         act_cfg=dict(type='GELU')
        
#         kernel_size = 5
#         self.flow_make = Sequential(
#             nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
#             nn.InstanceNorm2d(in_channels*2),
#             nn.GELU(),
#             nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False),
#         )

#     def forward(self, x1, x2, fusion_policy=None):
#         """Forward function."""

#         output = torch.cat([x1, x2], dim=1)
#         flow = self.flow_make(output)
#         f1, f2 = torch.chunk(flow, 2, dim=1)
#         x1_feat = self.warp(x1, f1) - x2
#         x2_feat = self.warp(x2, f2) - x1
        
#         if fusion_policy == None:
#             return x1_feat, x2_feat
        
#         output = FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)
#         return output

#     @staticmethod
#     def warp(x, flow):
#         n, c, h, w = x.size()

#         norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
#         col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
#         row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
#         grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
#         grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
#         grid = grid + flow.permute(0, 2, 3, 1) / norm

#         output = F.grid_sample(x, grid, align_corners=True)
#         return output


# class MixFFN(BaseModule):
#     """An implementation of MixFFN of Segformer. \
#         Here MixFFN is uesd as projection head of Changer.
#     Args:
#         embed_dims (int): The feature dimension. Same as
#             `MultiheadAttention`. Defaults: 256.
#         feedforward_channels (int): The hidden dimension of FFNs.
#             Defaults: 1024.
#         act_cfg (dict, optional): The activation config for FFNs.
#             Default: dict(type='ReLU')
#         ffn_drop (float, optional): Probability of an element to be
#             zeroed in FFN. Default 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#     """

#     def __init__(self,
#                  embed_dims,
#                  feedforward_channels,
#                  act_cfg=dict(type='GELU'),
#                  ffn_drop=0.,
#                  dropout_layer=None,
#                  init_cfg=None):
#         super(MixFFN, self).__init__(init_cfg)

#         self.embed_dims = embed_dims
#         self.feedforward_channels = feedforward_channels
#         self.act_cfg = act_cfg
#         self.activate = build_activation_layer(act_cfg)

#         in_channels = embed_dims
#         fc1 = Conv2d(
#             in_channels=in_channels,
#             out_channels=feedforward_channels,
#             kernel_size=1,
#             stride=1,
#             bias=True)
#         # 3x3 depth wise conv to provide positional encode information
#         pe_conv = Conv2d(
#             in_channels=feedforward_channels,
#             out_channels=feedforward_channels,
#             kernel_size=3,
#             stride=1,
#             padding=(3 - 1) // 2,
#             bias=True,
#             groups=feedforward_channels)
#         fc2 = Conv2d(
#             in_channels=feedforward_channels,
#             out_channels=in_channels,
#             kernel_size=1,
#             stride=1,
#             bias=True)
#         drop = nn.Dropout(ffn_drop)
#         layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
#         self.layers = Sequential(*layers)
#         self.dropout_layer = build_dropout(
#             dropout_layer) if dropout_layer else torch.nn.Identity()

#     def forward(self, x, identity=None):
#         out = self.layers(x)
#         if identity is None:
#             identity = x
#         return identity + self.dropout_layer(out)


# @MODELS.register_module()
# class Changer(BaseDecodeHead):
#     """The Head of Changer.

#     This head is the implementation of
#     `Changer <https://arxiv.org/abs/2209.08290>` _.

#     Args:
#         interpolate_mode: The interpolate mode of MLP head upsample operation.
#             Default: 'bilinear'.
#     """

#     def __init__(self, interpolate_mode='bilinear', **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)

#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)
#         assert num_inputs == len(self.in_index)

#         self.convs = nn.ModuleList()
#         for i in range(num_inputs):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))

#         self.fusion_conv = ConvModule(
#             in_channels=self.channels * num_inputs,
#             out_channels=self.channels // 2,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)
        
#         self.neck_layer = FDAF(in_channels=self.channels // 2)
        
#         # projection head
#         self.discriminator = MixFFN(
#             embed_dims=self.channels,
#             feedforward_channels=self.channels,
#             ffn_drop=0.,
#             dropout_layer=dict(type='DropPath', drop_prob=0.),
#             act_cfg=dict(type='GELU'))
                
#     def base_forward(self, inputs):
#         outs = []
#         for idx in range(len(inputs)):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             outs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners))

#         out = self.fusion_conv(torch.cat(outs, dim=1))
        
#         return out

#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         inputs1 = []
#         inputs2 = []
#         for input in inputs:
#             f1, f2 = torch.chunk(input, 2, dim=1)
#             inputs1.append(f1)
#             inputs2.append(f2)
        
#         out1 = self.base_forward(inputs1)
#         out2 = self.base_forward(inputs2)
#         out = self.neck_layer(out1, out2, 'concat')

#         out = self.discriminator(out)
#         out = self.cls_seg(out)

#         return out
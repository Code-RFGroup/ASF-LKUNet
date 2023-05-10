
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
import os


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x




class Inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv =  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, x):

        out = self.inconv(x)
        return out

class Conv_residual(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1),
            )
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=1, padding=1),
        
        )
    def forward(self, x):
        residual = x
        # print(x.shape)
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out

class en_dwconv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_dwconv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1),
            )
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels,kernel_size=7, padding=3, groups=in_channels),
        )
    def forward(self, x):
        residual = x
        # print(x.shape)
        out =  self.double_dwconv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out

class dw_grn_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.dwconv = nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels)
        self.grn = GRN(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1)
    def forward(self, x):

        x = self.norm(x)
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = self.grn(x)
        x = x.permute(0,3,1,2)

        x = self.conv(x)
        return x



class encoder_dw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            en_dwconv(in_channels, out_channels),
        )

    def forward(self, x):
        out = self.down(x)
        return out

class merge_c2f_d(nn.Module):
    def __init__(self, in_channels,out_channels,type,Conv):
        super().__init__()
        self.type = type
        if self.type == 'down':
            self.conv_change = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0)
        elif self.type == 'up':
            self.conv_change = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.dwconv_se = dw_grn_conv(out_channels * 2,out_channels)

    def forward(self, x1, x2):
        x2 = self.conv_change(x2)
        merge = torch.cat([x1,x2], dim=1)
        out = self.dwconv_se(merge)
        return out


class merge_c2f_t(nn.Module):
    def __init__(self, down_channels,up_channels,out_channels,Conv):
        super().__init__()

        self.conv_down = nn.Conv2d(down_channels, out_channels, kernel_size=2, stride=2,padding=0)
        self.conv_up = nn.ConvTranspose2d(up_channels, out_channels, kernel_size=2, stride=2)

        self.dwconv_se = dw_grn_conv(out_channels * 3,out_channels)

    def forward(self, x1, x2,x3):
        x1 = self.conv_down(x1)
        x3 = self.conv_up(x3)
        # assert x1.shape[1] == x2.shape[1] ,'shape no same'
        merge = torch.cat([x1, x2, x3], dim=1)

        out = self.dwconv_se(merge)
        return out

class de_up(nn.Module):

    def __init__(self, in_channels,up_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.upconv = Conv_residual(up_channels,out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        out = torch.cat([x2, x1], dim=1)

        out = self.upconv(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.conv_res = Conv_residual(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.conv(x)
        return x



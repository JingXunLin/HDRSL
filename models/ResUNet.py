# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:30:02 2023

@author: 34677
"""

import torch.nn as nn
from torch import cat as cat
import torch
import torch.nn.functional as F
from .Attention_module import CBAMBlock, SpatialAttention_WH, ChannelAttention_WH

' Branch0 block '
class Branch0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch0, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bt0 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bt0(x0)
        return x0

' Branch1 block '
class Branch1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)#图像的尺寸不变
        self.bt1 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bt1(x1)
        return x1

' Branch2 block '
class Branch2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch2, self).__init__()
        self.conv2_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_1 = nn.BatchNorm2d(out_ch)
        self.rl2_1 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x2 = self.conv2_1(x)
        x2 = self.bt2_1(x2)
        x2 = self.rl2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bt2_2(x2)
        return x2

' Branch3 block '
class Branch3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch3, self).__init__()
        self.conv3_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_1 = nn.BatchNorm2d(out_ch)
        self.rl3_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_2 = nn.BatchNorm2d(out_ch)
        self.rl3_2 = nn.LeakyReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_3 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x3 = self.conv3_1(x)
        x3 = self.bt3_1(x3)
        x3 = self.rl3_1(x3)
        x3 = self.conv3_2(x3)
        x3 = self.bt3_2(x3)
        x3 = self.rl3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.bt3_3(x3)
        return x3

' Branch4 block '
class Branch4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch4, self).__init__()
        self.conv4_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_1 = nn.BatchNorm2d(out_ch)
        self.rl4_1 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_2 = nn.BatchNorm2d(out_ch)
        self.rl4_2 = nn.LeakyReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_3 = nn.BatchNorm2d(out_ch)
        self.rl4_3 = nn.LeakyReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_4 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x4 = self.conv4_1(x)
        x4 = self.bt4_1(x4)
        x4 = self.rl4_1(x4)
        x4 = self.conv4_2(x4)
        x4 = self.bt4_2(x4)
        x4 = self.rl4_2(x4)
        x4 = self.conv4_3(x4)
        x4 = self.bt4_3(x4)
        x4 = self.rl4_3(x4)
        x4 = self.conv4_4(x4)
        x4 = self.bt4_4(x4)
        return x4

' Residual block with Inception module '
class ResB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResB, self).__init__()
        self.branch0 = Branch0(in_ch, out_ch)
        self.branch1 = Branch1(in_ch, out_ch // 4)
        self.branch2 = Branch2(in_ch, out_ch // 4)
        self.branch3 = Branch3(in_ch, out_ch // 4)
        self.branch4 = Branch4(in_ch, out_ch // 4)
        self.rl = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = cat((x1, x2, x3, x4), dim=1)
        x6 =  x0 + x5
        x7= self.rl(x6)
        return  x7

' Downsampling block '
class DownB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownB, self).__init__()
        self.res = ResB(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x1 = self.res(x)
        x2 = self.pool(x1)
        return x2, x1

' Upsampling block '
class UpB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpB, self).__init__()
        self.up = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding = 1, output_padding = 1 )
        self.res = ResB(out_ch*2, out_ch)
    def forward(self, x, x_):
        x1 = self.up(x)
        x2 = cat((x1 , x_), dim=1)
        x3 = self.res(x2)
        return x3

' Output layer '#将最后一层变成融合层
class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
       # self.Sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.conv(x)
        return x1

' Architecture of Res-UNet '
class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResUNet, self).__init__()
        self.down1 = DownB(in_channels, 8) 
        self.down12 = DownB(8, 64)
        self.down2 = DownB(64, 128)
        self.down3 = DownB(128, 256)
        self.down4 = DownB(256, 512)
        self.res = ResB(512, 1024)  

        self.up1 = UpB(1024, 512)
        self.up2 = UpB(512, 256)
        self.up3 = UpB(256, 128)
        self.up4 = UpB(128, 64)
        self.up34 = UpB(64, 8)
        self.outc = Outconv(8, out_channels)##返回2个channel,channel1是主路depth,channel2是支路1,wrapped_low

    def forward(self, x):
        x1, x1_ = self.down1(x)     # HW 8    
        x12, x12_ = self.down12(x1) # HW/2 64
        x2, x2_ = self.down2(x12)   # HW/4 128
        x3, x3_ = self.down3(x2)    # HW/8 256
        x4, x4_ = self.down4(x3)    # HW/16 512
        x5  = self.res(x4)          # X5 = X4  HW/32 1024

        x6  = self.up1(x5, x4_)     # HW/16 512 
        x7  = self.up2(x6, x3_)     # HW/8 256
        x8  = self.up3(x7, x2_)     # HW/4 128
        x9  = self.up4(x8, x12_)    # HW/2 64
        x9_10  = self.up34(x9, x1_) # HW 8
        x10 = self.outc(x9_10)      # HW 2

        return  x10, [x1, x2, x4, x7, x9]

class ResUNet_attention(ResUNet):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResUNet_attention, self).__init__(in_channels, out_channels)
        self.attention_encoder1 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        self.attention_encoder2 = CBAMBlock(channel=128, reduction=4, kernel_size=3)
        self.attention_decoder1 = CBAMBlock(channel=128, reduction=4, kernel_size=3)
        self.attention_decoder2 = CBAMBlock(channel=64, reduction=2, kernel_size=3)

    def forward(self, x):
        x1, x1_ = self.down1(x)     # HW 8    
        x12, x12_ = self.down12(x1) # HW/2 64
        x12 = self.attention_encoder1(x12)
        x2, x2_ = self.down2(x12)   # HW/4 128
        x2 = self.attention_encoder2(x2)
        x3, x3_ = self.down3(x2)    # HW/8 256
        x4, x4_ = self.down4(x3)    # HW/16 512
        x5  = self.res(x4)          # X5 = X4  HW/32 1024

        x6  = self.up1(x5, x4_)     # HW/16 512 
        x7  = self.up2(x6, x3_)     # HW/8 256
        x8  = self.up3(x7, x2_)     # HW/4 128
        x8 = self.attention_decoder1(x8)
        x9  = self.up4(x8, x12_)    # HW/2 64
        x9 = self.attention_decoder2(x9)
        x9_10  = self.up34(x9, x1_) # HW 8
        x10 = self.outc(x9_10)      # HW 2
        return  x10, [x1, x2, x4, x7, x9]


if __name__ == "__main__":
    x = torch.rand(2, 1, 512, 640)
    net = ResUNet()
    y = net(x)
    print(y.size())
#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.base import BaseNet
from model.modules.attention import SelfAttention
from model.modules.block import DownBlock, UpBlock
from model.modules.conv import DoubleConv

class DoubleConvPerso(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, norm_layer, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
        
        if norm_layer == "bn" :
            
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        elif norm_layer == "in" :
            
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvPerso(in_channels, out_channels, norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer, bilinear=True, skip_connection=True):
        super().__init__()
        
        self.skip_connection = skip_connection

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvPerso(in_channels, out_channels, in_channels // 2, norm_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvPerso(in_channels, out_channels, norm_layer)

    def forward(self, x1, x2):
        
        if self.skip_connection :
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        
        else :
            x1 = self.up(x1)
            return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
class UNet(nn.Module):
    def __init__(self, in_channel=3, n_classes=1, norm_layer="bn", bilinear=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm_layer = norm_layer

        self.inc = DoubleConvPerso(in_channel, 64, norm_layer)
        self.down1 = Down(64, 128, norm_layer)
        self.down2 = Down(128, 256, norm_layer)
        self.down3 = Down(256, 512, norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_layer)
        self.up1 = Up(1024, 512 // factor, norm_layer, bilinear)
        self.up2 = Up(512, 256 // factor, norm_layer, bilinear)
        self.up3 = Up(256, 128 // factor, norm_layer, bilinear)
        self.up4 = Up(128, 64, norm_layer, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return logits

class YNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(YNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear, skip_connection=False)
        self.up2 = Up(512, 256 // factor, bilinear, skip_connection=False)
        self.up3 = Up(256, 128 // factor, bilinear, skip_connection=False)
        self.up4 = Up(128, 64, bilinear, skip_connection=False)
        self.outc = OutConv(64, n_channels)
        
    def forward(self,x):
        x1 = self.up1(x,x)
        x2 = self.up2(x1,x1)
        x3 = self.up3(x2,x2)
        x4 = self.up4(x3,x3)
        return self.outc(x4)

class UNetAttention(BaseNet):
    """
    UNet
    """

    def __init__(self, in_channel=3, out_channel=1, channel=[3, 32, 64, 128, 256], time_channel=256, num_classes=None, image_size=64,
                 device="cpu", act="silu"):
        """
        Initialize the UNet network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super().__init__(in_channel, out_channel, channel, time_channel, num_classes, image_size, device, act)

        # channel: 3 -> 64
        # size: size
        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

        # channel: 64 -> 128
        # size: size / 2
        self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa1 = SelfAttention(channels=self.channel[2], size=int(self.image_size / 2), act=self.act)
        # channel: 128 -> 256
        # size: size / 4
        self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 4
        self.sa2 = SelfAttention(channels=self.channel[3], size=int(self.image_size / 4), act=self.act)
        # channel: 256 -> 256
        # size: size / 8
        self.down3 = DownBlock(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = SelfAttention(channels=self.channel[3], size=int(self.image_size / 8), act=self.act)

        # channel: 256 -> 512
        # size: size / 8
        self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 512
        # size: size / 8
        self.bot2 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 256
        # size: size / 8
        self.bot3 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[3], act=self.act)

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlock(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 4
        self.sa4 = SelfAttention(channels=self.channel[2], size=int(self.image_size / 4), act=self.act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 2
        self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa5 = SelfAttention(channels=self.channel[1], size=int(self.image_size / 2), act=self.act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size
        self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size
        self.sa6 = SelfAttention(channels=self.channel[1], size=int(self.image_size), act=self.act)

        # channel: 64 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, y=None):
        """
        Forward
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: output
        """
        #time = time.unsqueeze(-1).type(torch.float)
        #time = self.pos_encoding(time, self.time_channel)

        #if y is not None:
        #    time += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2_sa = self.sa1(x2)
        x3 = self.down2(x2_sa)
        x3_sa = self.sa2(x3)
        x4 = self.down3(x3_sa)
        x4_sa = self.sa3(x4)

        bot1_out = self.bot1(x4_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)

        up1_out = self.up1(bot3_out, x3_sa)
        up1_sa_out = self.sa4(up1_out)
        up2_out = self.up2(up1_sa_out, x2_sa)
        up2_sa_out = self.sa5(up2_out)
        up3_out = self.up3(up2_sa_out, x1)
        up3_sa_out = self.sa6(up3_out)
        output = self.outc(up3_sa_out)
        return output


if __name__ == "__main__":
    # Unconditional
    net = UNet(device="cpu", image_size=128)
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 128, 128)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    # print(net(x, t, y).shape)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:25:13 2020

@author: tycoer
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features

class FuseStage(nn.Module):
    def __init__(self, 
                 in_channels,
                 is_reversal = False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        super(FuseStage, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if is_reversal:
            self.right_conv = None
            self.left_conv = ConvModule(in_channels, in_channels * 2, kernel_size=3, stride=2, padding = 1, **cfg)
        else:
            self.right_conv = nn.Sequential(ConvModule(in_channels, in_channels // 2, kernel_size = 1, stride = 1,**cfg),
                                            nn.Upsample(scale_factor=2, mode='nearest'))
            self.left_conv = ConvModule(in_channels, in_channels // 2, kernel_size = 1, stride = 1,**cfg)
    def forward(self, data):
        left, right = data
        left = self.left_conv(left)
        if self.right_conv:
            right = self.right_conv(right)
        return torch.cat((left, right), axis = 1)


def MakeNConv(in_channels,
              out_channels,
              n,
              conv_cfg=None,
              norm_cfg=dict(type='BN', requires_grad=True),
              act_cfg=dict(type='Mish')):
    
    cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
    double_out_channels = out_channels * 2 
    if n == 3:
        m = nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
            ConvModule(out_channels,double_out_channels, kernel_size = 3, padding = 1,**cfg),
            ConvModule(double_out_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
        )
    elif n == 5:
        m = nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
            ConvModule(out_channels, double_out_channels, kernel_size = 3, padding = 1,**cfg),
            ConvModule(double_out_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
            ConvModule(out_channels, double_out_channels, kernel_size = 3, padding = 1,**cfg),
            ConvModule(double_out_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
        )
    else:
        raise NotImplementedError
    return m



class YOLOV4Neck(nn.Module):
    
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU')):
        super(YOLOV4Neck, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.layers = nn.ModuleList([
            # SPP
            nn.Sequential(MakeNConv(1024, 512, 3,**cfg,),
                          SpatialPyramidPooling(),
                          MakeNConv(2048, 512, 3,**cfg)),
            # PANet
            nn.Sequential(FuseStage(512, **cfg),
                          MakeNConv(512, 256, 5,**cfg)),
             
            nn.Sequential(FuseStage(256, **cfg),
                          MakeNConv(256, 128, 5,**cfg)),
             
            nn.Sequential(FuseStage(128, **cfg, is_reversal=True),
                          MakeNConv(512,256, 5, **cfg)),
             
            nn.Sequential(FuseStage(256, **cfg, is_reversal=True),
                          MakeNConv(1024,512, 5,**cfg))
            ])
        
    def forward(self, x):
        out3, out4, out5 = x
        out5 = self.layers[0](out5)
        out4 = self.layers[1]([out4, out5])

        out3 = self.layers[2]([out3, out4])  # 输出0 大图
        out4 = self.layers[3]([out3, out4])  # 输出1
        out5 = self.layers[4]([out4, out5])  # 输出2 小图

        return (out5, out4, out3)

    def init_weights(self):
        pass

                
                
if __name__ == '__main__':
    import numpy as np
    from cspdarknet import CSPDarknet

    img = np.zeros((1,3,416,416),dtype = 'float32')
    tensor = torch.from_numpy(img)
    backbone = CSPDarknet()
    neck = YOLOV4Neck()
    
    outs_backbone = backbone.forward(tensor)
    for out in outs_backbone:
        print(out.size())
    outs_neck = neck.forward(outs_backbone)
    for out in outs_neck:
        print(out.size())

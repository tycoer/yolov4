# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:25:13 2020

@author: tycoer


本文件用于将 Darknet官方提供的 .weight文件 转成 .pth文件

"""



                
if __name__ == '__main__':
    import numpy as np
    from yolo4_backbone1 import CSPDarknet
    from yolo4_neck1 import YOLOV4Neck
    from collections import OrderedDict
    import torch
    from mmcv.cnn import ConvModule
    import torch.nn as nn 
    
    #加入head部分的参数
    class YOLOV3Head(nn.Module):
        def __init__(self,
                      in_channels=[512, 256, 128],
                      out_channels=(1024, 512, 256),
                      num_anchors = 3,
                      num_class = 80,
                      conv_cfg=None,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      act_cfg=dict(type='ReLU')
                      ):
            
            
            cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            super().__init__()
            self.convs_bridge = nn.ModuleList()
            self.convs_pred = nn.ModuleList()
            for i in range(3):
                conv_bridge = ConvModule(in_channels[i],out_channels[i],3,padding=1, **cfg)
                conv_pred = nn.Conv2d(out_channels[i],num_anchors * (5 + num_class), 1)

                self.convs_bridge.append(conv_bridge)
                self.convs_pred.append(conv_pred)
                
    head_dict = YOLOV3Head().state_dict()
    backbone_dict = CSPDarknet().state_dict()
    neck_dict = YOLOV4Neck().state_dict()
    
    
    
    # 读取官方的权重文件
    filename = 'F:/project/mmtest/weight/yolov4-mish.weights'
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)  # 前5个是文件头
        elements = np.fromfile(f, dtype=np.float32)  # 剩下的就是参数了
    
    
    

    ######################## 排序 neck + head #####################
    '''
    由于 在 官方的网络中 neck 和 head 部分的网络是穿插在一起的, 为以下结构:
        neck的out3;
        head的ConvModule + Conv2d
        neck的out4;
        head的ConvModule + Conv2d
        neck的out5;
        head的ConvModule + Conv2d
        
        
    故下列语句将本人实现的 neck 和 head 如 官方结构进行排序
    '''
    
    def sortConvModule(keys, values):
        assert len(keys) == len(values)
        net_sorted_dict = OrderedDict()
        
        '''
        !!!!!!卷积(conv + bn + act)的结构!!!!!!!
        
        
        .weight文件中卷积参数的顺序     torch 中的卷积参数顺序
        0 bn.bias                     0 conv.weight
        1 bn.weight                   1 bn.weight
        2 bn.running_mean             2 bn.bias
        3 bn.running_var              3 bn.running_mean
        4 conv.weight                 4 bn.running_var
                                      5 bn.num_batches_tracked (.weight中没有该参数, 应舍去)
        
        
        故 本函数功能为: 将torch中的顺序 转为.weight中的顺序
        '''
        
        for i in range(0,len(keys),6):
            net_sorted_dict[keys[2+i]] = values[2+i]
            net_sorted_dict[keys[1+i]] = values[1+i]
            net_sorted_dict[keys[3+i]] = values[3+i]
            net_sorted_dict[keys[4+i]] = values[4+i]
            net_sorted_dict[keys[0+i]] = values[0+i]
        return net_sorted_dict
    
    def sortConv2d(keys,values):
        '''
        !!!!!!卷积(conv)的结构!!!!!!!
        
        
        .weight文件中卷积参数的顺序     torch 中的卷积参数顺序
        0 conv.bias                  0 conv.weight
        1 conv.weight                1 conv.bias  
        '''
        assert len(keys) == len(values)
        net_sorted_dict = OrderedDict()
        for i in range(0, len(keys),2):
            net_sorted_dict[keys[1+i]] = values[1+i]
            net_sorted_dict[keys[0+i]] = values[0+i]
        return net_sorted_dict
    
    neck_head_sorted_dict = OrderedDict()
    
    neck_keys,neck_values = list(neck_dict.keys()), list(neck_dict.values())
    head_keys,head_values = list(head_dict.keys()), list(head_dict.values())
    
    
    neck_head_sorted_dict.update(sortConvModule(neck_keys[:108], neck_values[:108]))
    neck_head_sorted_dict.update(sortConvModule(head_keys[:6],head_values[:6]))
    neck_head_sorted_dict.update(sortConv2d(head_keys[18:20], head_values[18:20]))

    neck_head_sorted_dict.update(sortConvModule(neck_keys[108:138], neck_values[108:138]))
    neck_head_sorted_dict.update(sortConvModule(head_keys[6:12],head_values[6:12]))
    neck_head_sorted_dict.update(sortConv2d(head_keys[20:22], head_values[20:22]))

    neck_head_sorted_dict.update(sortConvModule(neck_keys[138:], neck_values[138:]))
    neck_head_sorted_dict.update(sortConvModule(head_keys[12:18],head_values[12:18]))
    neck_head_sorted_dict.update(sortConv2d(head_keys[22:], head_values[22:]))
    
    
    ################################## backbone ##############################
    backbone_sorted_dict = OrderedDict()
    backbone_keys, backbone_values = tuple(backbone_dict.keys()),tuple(backbone_dict.values())
    backbone_sorted_dict = sortConvModule(backbone_keys, backbone_values)
    
    
    #####################= 合并 backbone 和 neck + head ############
    backbone_neck_head_dict = OrderedDict()
    backbone_neck_head_dict.update(backbone_sorted_dict)
    backbone_neck_head_dict.update(neck_head_sorted_dict)
    
    
    
    start = 0
    new_dict = OrderedDict()
    
    for key, value in backbone_sorted_dict.items():
        # print(key)
        # total_element += value.numel()
        new_dict[key] = torch.from_numpy(elements[start:start + value.numel()]).view_as(value)
        start += value.numel()
    new_dict = {"state_dict": new_dict}   
    torch.save(new_dict, 'F:/project/mmtest/weight/yolov4_hy.pth')
    
    

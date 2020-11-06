# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:22:17 2020

@author: tycoer
"""

import argparse
import torch

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection image demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('imagepath', help='the path of image to test')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    result = inference_detector(model, args.imagepath)
    # 这里的result是一个列表，长度为类别数，例如我这里就是6
    # 其中每个元素就是对一类的预测出来的bbox，是一个np.ndarray
    # shape为(N,5),N可能大于测试图中实际的该类的数量
    # 5是4个坐标值，加1个置信度
    show_result(
        args.imagepath, result, model.CLASSES, score_thr=args.score_thr, wait_time=0)


if __name__ == '__main__':
    main()

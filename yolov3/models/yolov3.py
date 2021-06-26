from collections import defaultdict

import torch
from torch import nn as nn

from yolov3.models.yolo_layer import YOLOLayer


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a Conv2d / BatchNorm2d / leaky ReLU block.

    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    pad = (ksize - 1) // 2

    sequential = nn.Sequential()
    sequential.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=False,
        ),
    )
    sequential.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    sequential.add_module("leaky", nn.LeakyReLU(0.1))

    return sequential


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of two convolution layers.

    Args:
        ch (int): number of input and output channels.
        n_blocks (int): number of residual blocks.
    """

    def __init__(self, ch, n_blocks):
        super().__init__()

        self.module_list = nn.ModuleList()
        for i in range(n_blocks):
            resblock = nn.ModuleList(
                [
                    add_conv(in_ch=ch, out_ch=ch // 2, ksize=1, stride=1),
                    add_conv(in_ch=ch // 2, out_ch=ch, ksize=3, stride=1),
                ]
            )
            self.module_list.append(resblock)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h

        return x


def create_yolov3_modules(config_model):
    # layer order is same as yolov3.cfg
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    module_list = nn.ModuleList()

    #
    # Darknet 53
    #

    module_list.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))  # 0 / 0
    module_list.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))  # 1 / 1
    # 1
    module_list.append(resblock(ch=64, n_blocks=1))  # 2 ~ 4 / 2
    module_list.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))  # 5 / 3
    # 2
    module_list.append(resblock(ch=128, n_blocks=2))  # 6 ~ 11 / 4
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))  # 12 / 5
    # 3
    module_list.append(resblock(ch=256, n_blocks=8))  # 13 ~ 36 / 6
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))  # 37 / 7
    # 4
    module_list.append(resblock(ch=512, n_blocks=8))  # 38 ~ 61 / 8
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))  # 62 / 9
    # 5
    module_list.append(resblock(ch=1024, n_blocks=4))  # 63 ~ 74 / 10

    #
    # additional layers for YOLOv3
    #

    # A
    module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 75 / 11
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 76 / 12
    module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 77 / 13
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 78 / 14
    module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 79 / 15
    # B
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 80 / 16
    module_list.append(YOLOLayer(config_model, layer_no=0, in_ch=1024))  # 81, 82 / 17

    # path 83 / 15 -> 18

    # C
    module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 84 / 18
    module_list.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 85 / 19

    # concat 86 / 19 (128) + 8 (512) = 20 (768)

    # A
    module_list.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))  # 87 / 20
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 88 / 21
    module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 89 / 22
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 90 / 23
    module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 91 / 24
    # B
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 92 / 25
    module_list.append(YOLOLayer(config_model, layer_no=1, in_ch=512))  # 93, 94 / 26

    # path 95 / 24 -> 27

    # C
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 96 / 27
    module_list.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 97 / 28

    # concat 28 (128) + 6 (256) = 20 (384)

    # A
    module_list.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))  # 98 / 29
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 99 / 30
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 100 / 31
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 101 / 32
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 102 / 33
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 103 / 34
    module_list.append(YOLOLayer(config_model, layer_no=2, in_ch=256))  # 105, 106 / 35

    return module_list


class YOLOv3(nn.Module):
    def __init__(self, config_model):
        super().__init__()
        self.module_list = create_yolov3_modules(config_model)

    def forward(self, x, labels=None):
        train = labels is not None
        self.loss_dict = defaultdict(float)

        output = []
        layers = []
        for i, module in enumerate(self.module_list):

            if i == 18:
                x = layers[i - 3]
            if i == 20:
                x = torch.cat((layers[i - 1], layers[8]), dim=1)
            if i == 27:
                x = layers[i - 3]
            if i == 29:
                x = torch.cat((layers[i - 1], layers[6]), dim=1)

            if isinstance(module, YOLOLayer):
                if train:
                    x, *losses = module(x, labels)
                    for name, loss in zip(["xy", "wh", "obj", "cls"], losses):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)

                output.append(x)
            else:
                x = module(x)

            layers.append(x)

        if train:
            return sum(output)
        else:
            return torch.cat(output, dim=1)


def create_yolov3_tiny_modules(config_model):
    # layer order is same as yolov3-tiny.cfg
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
    module_list = nn.ModuleList()

    #
    # Darknet 14
    #

    module_list.append(add_conv(in_ch=3, out_ch=16, ksize=3, stride=1))  # 0 / 0
    module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 1 / 1

    module_list.append(add_conv(in_ch=16, out_ch=32, ksize=3, stride=1))  # 2 / 2
    module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 3 / 3

    module_list.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=1))  # 4 / 4
    module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 5 / 5

    module_list.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=1))  # 6 / 6
    module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 7 / 7

    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 8 / 8
    module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 9 / 9

    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 10 / 10
    module_list.append(nn.ZeroPad2d((0, 1, 0, 1)))  # / 11
    module_list.append(nn.MaxPool2d(kernel_size=2, stride=1))  # 11 / 12

    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 12 / 13

    #
    # additional layers for YOLOv3
    #

    # A
    module_list.append(add_conv(in_ch=1024, out_ch=256, ksize=1, stride=1))  # 13 / 14
    # B
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 14 / 15
    module_list.append(YOLOLayer(config_model, layer_no=0, in_ch=512))  # 15, 16 / 16

    # path 17 / 14 -> 17

    # C
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 18 / 17
    module_list.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 19 / 18

    # concat 19 (128) + 8 (256) = 20 (384)

    # B
    module_list.append(add_conv(in_ch=384, out_ch=256, ksize=3, stride=1))  # 21 / 19
    module_list.append(YOLOLayer(config_model, layer_no=1, in_ch=256))  # 22, 23 / 20

    return module_list


class YOLOv3Tiny(nn.Module):
    def __init__(self, config_model):
        super().__init__()
        self.module_list = create_yolov3_tiny_modules(config_model)

    def forward(self, x, labels=None):
        train = labels is not None
        self.loss_dict = defaultdict(float)

        output = []
        layers = []
        for i, module in enumerate(self.module_list):
            if i == 17:
                x = layers[i - 3]
            if i == 19:
                x = torch.cat((layers[i - 1], layers[8]), dim=1)

            if isinstance(module, YOLOLayer):
                if train:
                    x, *losses = module(x, labels)
                    for name, loss in zip(["xy", "wh", "obj", "cls"], losses):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)

                output.append(x)
            else:
                x = module(x)

            layers.append(x)

        if train:
            return sum(output)
        else:
            return torch.cat(output, dim=1)

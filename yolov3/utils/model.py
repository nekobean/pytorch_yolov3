import numpy as np
import torch

from yolov3.models.yolov3 import YOLOv3, YOLOv3Tiny


def conv_initializer(param):
    out_ch, in_ch, h, w = param.shape
    fan_in = h * w * in_ch
    scale = np.sqrt(2 / fan_in)

    w = scale * torch.randn_like(param)

    # n = scale * np.random.normal(size=param.numel())
    # w = torch.from_numpy(n).view_as(param)

    return w


def parse_conv_block(module, weights, offset, initialize):
    """
    Initialization of conv layers with batchnorm
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initialize (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    # Darknet serializes weights of convolutional layer
    # with batch normalization as following order.
    # - bias: (out_ch,)
    # - gamma: (out_ch,)
    # - mean: (out_ch,)
    # - bias: (out_ch,)
    # - kernel: (out_ch, in_ch, h, w)
    conv, bn, leakey = module

    params = [
        bn.bias,
        bn.weight,
        bn.running_mean,
        bn.running_var,
        conv.weight,
    ]

    for param in params:
        if initialize:
            if param is bn.weight:
                w = torch.ones_like(param)
            elif param is conv.weight:
                w = conv_initializer(param)
            else:
                w = torch.zeros_like(param)
        else:
            param_len = param.numel()
            w = torch.from_numpy(weights[offset : offset + param_len]).view_as(param)
            offset += param_len

        param.data.copy_(w)

    return offset


def parse_yolo_block(module, weights, offset, initialize):
    """
    YOLO Layer (one conv with bias) Initialization
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initialize (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    # Darknet serializes weights of convolutional layer
    # without batch normalization as following order.
    # - bias: (out_ch,)
    # - kernel: (out_ch, in_ch, h, w)
    conv = module._modules["conv"]

    for param in [conv.bias, conv.weight]:
        if initialize:
            if param is conv.bias:
                w = torch.zeros_like(param)
            else:
                w = conv_initializer(param)
        else:
            param_len = param.numel()
            w = torch.from_numpy(weights[offset : offset + param_len]).view_as(param)
            offset += param_len

        param.data.copy_(w)

    return offset


def parse_yolo_weights(model, weights_path):
    """
    Parse YOLO (darknet) pre-trained weights data onto the pytorch model
    Args:
        model : pytorch model object
        weights_path (str): path to the YOLO (darknet) pre-trained weights file
    """
    with open(weights_path, "rb") as f:
        # skip header
        f.read(20)

        # read weights
        weights = np.fromfile(f, dtype=np.float32)

    offset = 0
    initialize = False

    for module in model.module_list:
        if module._get_name() == "Sequential":
            # conv / batch norm / leaky LeRU
            offset = parse_conv_block(module, weights, offset, initialize)

        elif module._get_name() == "resblock":
            # residual block
            for resblocks in module._modules["module_list"]:
                for resblock in resblocks:
                    offset = parse_conv_block(resblock, weights, offset, initialize)

        elif module._get_name() == "YOLOLayer":
            # YOLO Layer (one conv with bias) Initialization
            offset = parse_yolo_block(module, weights, offset, initialize)

        # the end of the weights file. turn the flag on
        initialize = offset >= len(weights)


def create_model(config):
    # モデルを作成する。
    if config["model"]["name"] == "yolov3":
        model = YOLOv3(config["model"])
    else:
        model = YOLOv3Tiny(config["model"])

    return model

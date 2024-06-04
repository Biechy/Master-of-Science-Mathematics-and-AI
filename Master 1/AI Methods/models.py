"""
Module to defined all the models of ResNet and classic NN with architecture wanted
"""

from classes import ResNetNN, ConvBlock

from torchvision import models as m
from torch import nn


def ResNet56A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[9, 9, 9], option="A", ResNet=True)


def ResNet44A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[7, 7, 7], option='A', ResNet=True)


def ResNet32A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[5, 5, 5], option='A', ResNet=True)


def ResNet20A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3, 3, 3], option='A', ResNet=True)


def ResNet56B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[9, 9, 9], option="B", ResNet=True)


def ResNet44B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[7, 7, 7], option='B', ResNet=True)


def ResNet32B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[5, 5, 5], option='B', ResNet=True)


def ResNet20B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3, 3, 3], option='B', ResNet=True)


def CNN56():
    return ResNetNN(
        block_type=ConvBlock, num_blocks=[9, 9, 9], option="A", ResNet=False
    )


def CNN44():
    return ResNetNN(block_type=ConvBlock, num_blocks=[7, 7, 7], option='A', ResNet=False)


def CNN32():
    return ResNetNN(block_type=ConvBlock, num_blocks=[5, 5, 5], option='A', ResNet=False)


def CNN20():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3, 3, 3], option='A', ResNet=False)


def TorchResNet50():
    model = m.resnet50(pretrained=False)
    model.fc = nn.LazyLinear(10)
    return model


def TorchResNet34():
    model = m.resnet34(pretrained=False)
    model.fc = nn.LazyLinear(10)
    return model


def TorchResNet18():
    model = m.resnet18(pretrained=False)
    model.fc = nn.LazyLinear(10)
    return model


def ResNet8():
    return ResNetNN(block_type=ConvBlock, num_blocks=[1, 1, 1], option='B', ResNet=True)


def ResNet14():
    return ResNetNN(block_type=ConvBlock, num_blocks=[2, 2, 2], option='B', ResNet=True)


def ResNet20():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3, 3, 3], option='B', ResNet=True)


def CNN8():
    return ResNetNN(block_type=ConvBlock, num_blocks=[1, 1, 1], option='A', ResNet=False)


def CNN14():
    return ResNetNN(block_type=ConvBlock, num_blocks=[2, 2, 2], option='A', ResNet=False)

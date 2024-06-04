"""
Module to define classes needed in ResNet Network architecture
"""
import torch
from torch import nn
import torch.nn.functional as F


# needed to return an tensor in option A
class Identity(nn.Module):
    def __init__(self, lambd):
        super(Identity, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# will be use as block_type next
class ConvBlock(nn.Module):
    """
    ConvBlock will implement the regular ConvBlock and the shortcut block. See figure 2
    When the dimension changes between 2 blocks, the option A or B is used.
    """

    def __init__(self, in_channels, out_channels, stride=1, option="A", ResNet=True):
        super(ConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()  # identity when in_channels=out_channels

        self.relu = nn.ReLU()

        """Implementation of option A (adding pad) and B (conv2d) in the paper for matching dimensions"""

        if stride != 1 or in_channels != out_channels:
            if option == "A":
                # number 4 as said in the paper (4pixels are padded each side)
                pad_to_add = out_channels // 4
                # padding the right and bottom of tensor
                padding = (0, 0, 0, 0, pad_to_add, pad_to_add, 0, 0)
                self.shortcut = Identity(lambda x: F.pad(x[:, :, ::2, ::2], padding))
            if option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )

        self.ResNet = ResNet

    def forward(self, x):
        out = self.features(x)
        # sum it up with shortcut layer
        if self.ResNet:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetNN(nn.Module):
    """ResNet global architecture for CIFAR-10 DataSet of shape 32*32*3"""

    def __init__(self, block_type, num_blocks, option, ResNet):
        super(ResNetNN, self).__init__()

        self.in_channels = 16

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.block1 = self.__build_layer(
            block_type,
            16,
            num_blocks[0],
            starting_stride=1,
            option=option,
            ResNet=ResNet,
        )
        self.block2 = self.__build_layer(
            block_type,
            32,
            num_blocks[1],
            starting_stride=2,
            option=option,
            ResNet=ResNet,
        )
        self.block3 = self.__build_layer(
            block_type,
            64,
            num_blocks[2],
            starting_stride=2,
            option=option,
            ResNet=ResNet,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64, 10)  # final activation for classification

    def __build_layer(
        self, block_type, out_channels, num_blocks, starting_stride, option, ResNet
    ):
        # create a list of len num_blocks with the first the stride we want then follow by ones
        strides_list_for_current_block = [starting_stride] + [1] * (num_blocks - 1)

        # loop to create a mutiple layer with de good in_channels and out_channels and the good stride defined above
        layers = []
        for stride in strides_list_for_current_block:
            layers.append(
                block_type(self.in_channels, out_channels, stride, option, ResNet)
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

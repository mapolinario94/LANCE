import torch.nn as nn
from collections import OrderedDict
import torch
from .basemodel import *
from .linear_lance_cl import Linear_LANCE_CL
from .conv_lance_cl import Conv2d_LANCE_CL

__all__ = ['ResNet18']


def conv3x3(in_planes, out_planes, stride=1, mode="IN", threshold=0.95, threshold_cl=0.98, num_free_dim=0):
    return Conv2d_LANCE_CL(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, threshold=threshold, threshold_cl=threshold_cl)


def conv7x7(in_planes, out_planes, stride=1, mode="IN", threshold=0.95, threshold_cl=0.98, num_free_dim=0):
    return Conv2d_LANCE_CL(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False, threshold=threshold, threshold_cl=threshold_cl)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, threshold=0.95, threshold_cl=0.98, num_free_dim=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Identity()
        self.bn3 = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Conv2d_LANCE_CL(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                                         bias=False, threshold=0.9, threshold_cl=0.98)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)

        self.count = 0

    def forward(self, x, experience_id=None):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.bn3(self.shortcut(x))
        out = torch.relu(out)
        return out

    def forward_all_layers(self, x0, experience_id=None):
        x1 = torch.relu(self.bn1(self.conv1(x0)))
        x2 = self.bn2(self.conv2(x1))
        x2 += self.bn3(self.shortcut(x0))
        x2 = torch.relu(x2)
        return x1, x2


class ResNet(BaseModel):
    def __init__(self, nf, n_classes=10, n_experiences=3, threshold_cl=0.98, threshold=0.95, stride_first=False, num_free_dim=0):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1 if stride_first else 2, num_free_dim=num_free_dim)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.block1 = BasicBlock(self.in_planes, nf * 1, stride=1, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 1 * BasicBlock.expansion
        self.block2 = BasicBlock(self.in_planes, nf * 1, stride=1, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 1 * BasicBlock.expansion
        self.block3 = BasicBlock(self.in_planes, nf * 2, stride=2, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 2 * BasicBlock.expansion
        self.block4 = BasicBlock(self.in_planes, nf * 2, stride=1, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 2 * BasicBlock.expansion
        self.block5 = BasicBlock(self.in_planes, nf * 4, stride=2, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 4 * BasicBlock.expansion
        self.block6 = BasicBlock(self.in_planes, nf * 4, stride=1, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 4 * BasicBlock.expansion
        self.block7 = BasicBlock(self.in_planes, nf * 8, stride=2, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 8 * BasicBlock.expansion
        self.block8 = BasicBlock(self.in_planes, nf * 8, stride=1, threshold=threshold, threshold_cl=threshold_cl, num_free_dim=num_free_dim)
        self.in_planes = nf * 8 * BasicBlock.expansion

        self.linear = torch.nn.ModuleList()
        for t in range(n_experiences):
            self.linear.append(nn.Linear(nf * 8 * BasicBlock.expansion * 4, n_classes, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, labels=None, experience_id=None):
        bsz = x.size(0)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = torch.nn.AdaptiveAvgPool2d((2,2))(out)
        out = out.view(out.size(0), -1)
        y_pred = self.linear[experience_id](out)
        return y_pred

    def forward_all_layers(self, x0, experience_id=None):
        bsz = x0.size(0)
        x1 = torch.relu(self.bn1(self.conv1(x0, experience_id)))
        x2, x3 = self.block1.forward_all_layers(x1, experience_id)
        x4, x5 = self.block2.forward_all_layers(x3, experience_id)
        x6, x7 = self.block3.forward_all_layers(x5, experience_id)
        x8, x9 = self.block4.forward_all_layers(x7, experience_id)
        x10, x11 = self.block5.forward_all_layers(x9, experience_id)
        x12, x13 = self.block6.forward_all_layers(x11, experience_id)
        x14, x15 = self.block7.forward_all_layers(x13, experience_id)
        x16, x17 = self.block8.forward_all_layers(x15, experience_id)
        x17 = torch.nn.AdaptiveAvgPool2d((2, 2))(x17)
        x17 = x17.view(x17.size(0), -1)
        return x0, x1, x2, x3, x4, x5, x5, x6, x7, x8, x9, x9, x10, x11, x12, x13, x13, x14, x15, x16, x17


def ResNet18(n_classes=10, n_experiences=3, nf=20, threshold_cl=0.95, threshold=0.95, stride_first=False, num_free_dim=0):
    return ResNet(nf=nf, n_classes=n_classes, n_experiences=n_experiences, threshold=threshold,
                  threshold_cl=threshold_cl, stride_first=stride_first, num_free_dim=num_free_dim)
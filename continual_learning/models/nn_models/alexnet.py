import torch
from torch import nn
from .basemodel import BaseModel
from .linear_lance_cl import Linear_LANCE_CL
from .conv_lance_cl import Conv2d_LANCE_CL

__all__ = ['AlexNet']


class AlexNet(BaseModel):

    def __init__(self, n_classes=10, n_experiences=3, threshold=0.90, threshold_cl=0.95, num_free_dim=0):
        super(AlexNet, self).__init__()

        self.conv1 = Conv2d_LANCE_CL(3, 64, 4, bias=False, activate=False, threshold=threshold, threshold_cl=threshold_cl)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv2 = Conv2d_LANCE_CL(64, 128, 3, bias=False, activate=False, threshold=threshold, threshold_cl=threshold_cl)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv3 = Conv2d_LANCE_CL(128, 256, 2, bias=False, activate=False, threshold=threshold, threshold_cl=threshold_cl)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)

        self.fc1 = Linear_LANCE_CL(256 * 4, 2048, bias=False, activate=False, threshold=threshold, threshold_cl=threshold_cl)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = Linear_LANCE_CL(2048, 2048, bias=False, activate=False, threshold=threshold, threshold_cl=threshold_cl)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)

        self.fc3 = torch.nn.ModuleList()
        for t in range(n_experiences):
            self.fc3.append(torch.nn.Linear(2048, n_classes, bias=False))

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.flag = False

    def forward(self, x, labels=None, experience_id=None):
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y_pred = self.fc3[experience_id](x)
        return y_pred

    def forward_all_layers(self, x0, experience_id=None):
        x = self.conv1(x0, task_id=experience_id)
        x1 = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        x = self.conv2(x1, task_id=experience_id)
        x2 = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        x = self.conv3(x2, task_id=experience_id)
        x3 = self.maxpool(self.drop2(self.relu(self.bn3(x))))
        x3 = x3.view(x3.size(0), -1)
        x = self.fc1(x3, task_id=experience_id)
        x4 = self.drop2(self.relu(self.bn4(x)))
        x = self.fc2(x4, task_id=experience_id)
        x5 = self.drop2(self.relu(self.bn5(x)))
        return x0, x1, x2, x3, x4, x5

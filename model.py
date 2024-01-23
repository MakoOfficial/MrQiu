import numpy as np
import pandas as pd
import os, sys, random
import numpy as np
import pandas as pd
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _utils

from random import choice

from skimage import io
from PIL import Image, ImageOps

import glob

# from torchsummary import summary
import logging

import matplotlib.pyplot as plt

import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")
from torchvision.models import resnet34, resnet50

seed = 1  # seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)  # numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False

import numpy as np
import pandas as pd
import os, sys, random
import numpy as np
import pandas as pd
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _utils

from random import choice

from skimage import io
from PIL import Image, ImageOps

import glob

# from torchsummary import summary
import logging

import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.utils import shuffle
# from apex import amp

import random

import time

from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter

# from albumentations.augmentations.transforms import Lambda, ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, RandomResizedCrop
# from albumentations.pytorch import ToTensor
# from albumentations import Compose, OneOrOther
#
# import warnings
# import torch_xla
# import torch_xla.debug.metrics as met
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.test.test_utils as test_utilsget_My_resnet34
import warnings

warnings.filterwarnings("ignore")
from pretrainedmodels import se_resnext101_32x4d, se_resnet152, xception, inceptionv4, inceptionresnetv2, inceptionv3
from torchvision.models import resnet34, resnet50, efficientnet_v2_s, mobilenet_v2


def get_My_resnet34():
    model = resnet34(pretrained=True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels


def get_My_resnet50(pretrained=True):
    model = resnet50(pretrained=pretrained)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels


def get_My_se_resnet152():
    model = se_resnet152(pretrained=None)
    output_channels = model.last_linear.in_features
    model = nn.Sequential(*list(model.children())[:-2])
    return model, output_channels


def get_My_se_resnext101_32x4d():
    model = se_resnext101_32x4d(pretrained=None)
    output_channels = model.last_linear.in_features
    model = nn.Sequential(*list(model.children())[:-2])
    return model, output_channels


def get_My_inceptionv4():
    model = inceptionv4(pretrained=None)
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]

    model = nn.Sequential(*model)
    return model, output_channels


def get_My_inceptionv3():
    model = inceptionv3(pretrained=None)
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-3]

    model = nn.Sequential(*model)
    return model, output_channels


def get_My_inceptionresnetv2():
    model = inceptionresnetv2(pretrained=None)
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]

    model = nn.Sequential(*model)
    return model, output_channels


def get_My_xception():
    model = xception(pretrained=None)
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]

    model = nn.Sequential(*model)
    return model, output_channels


def get_My_efficientnetv2():
    model = efficientnet_v2_s(weights=None)
    # output_channels = model.last_linear.in_features
    output_channels = model.classifier[1].in_features
    model = list(model.children())[:-2]

    model = nn.Sequential(*model)
    return model, output_channels


def get_My_mobilenetv2():
    model = mobilenet_v2(weights=None)
    output_channels = model.last_channel
    model = list(model.children())[:-1]

    model = nn.Sequential(*model)
    return model, output_channels


class baseline(nn.Module):

    def __init__(self, gender_length, backbone, out_channels) -> None:
        super(baseline, self).__init__()
        self.backbone = nn.Sequential(*backbone)
        self.out_channels = out_channels

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, gender_length),
            nn.BatchNorm1d(gender_length),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(in_features=out_channels + gender_length, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(512, 230, bias=False)
        self.to_latent = nn.Linear(512, 512, bias=False)

    def forward(self, x, gender):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)

        gender_encode = self.gender_encoder(gender)

        features = self.MLP(torch.cat((x, gender_encode), dim=-1))
        return self.to_latent(features), self.classifier(features)

class Res50Align(nn.Module):

    def __init__(self, gender_length, backbone, out_channels) -> None:
        super(Res50Align, self).__init__()
        self.backbone = nn.Sequential(*backbone)
        self.out_channels = out_channels

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, gender_length),
            nn.BatchNorm1d(gender_length),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(in_features=out_channels + gender_length, out_features=1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 1024, bias=False),
            # nn.BatchNorm1d(2048, affine=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 230, bias=False)
        )

    def forward(self, x, gender):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)
        gender_encode = self.gender_encoder(gender)

        logits = self.MLP(torch.cat((x, gender_encode), dim=-1))

        return logits, self.classifier(logits)


if __name__ == '__main__':
    res50 = baseline(32, *get_My_resnet50())
    res34 = baseline(32, *get_My_resnet34())
    mbnet = baseline(32, *get_My_mobilenetv2())
    effiNet = baseline(32, *get_My_efficientnetv2())
    xceptNet = baseline(32, *get_My_xception())
    inceptNetv3 = baseline(32, *get_My_inceptionv3())
    inceptNetv4 = baseline(32, *get_My_inceptionv4())
    inceptRes = baseline(32, *get_My_inceptionresnetv2())
    se_res152 = baseline(32, *get_My_se_resnet152())
    se_resnext = baseline(32, *get_My_se_resnext101_32x4d())

    print(f'res50:{sum(p.nelement() for p in res50.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'res34:{sum(p.nelement() for p in res34.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'mbnet:{sum(p.nelement() for p in mbnet.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'effiNet:{sum(p.nelement() for p in effiNet.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'xceptNet:{sum(p.nelement() for p in xceptNet.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'inceptNetv3:{sum(p.nelement() for p in inceptNetv3.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'inceptNetv4:{sum(p.nelement() for p in inceptNetv4.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'inceptRes:{sum(p.nelement() for p in inceptRes.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'se_res152:{sum(p.nelement() for p in se_res152.parameters() if p.requires_grad == True) / 1e6}M')
    print(f'se_resnext:{sum(p.nelement() for p in se_resnext.parameters() if p.requires_grad == True) / 1e6}M')


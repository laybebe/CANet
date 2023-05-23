from functools import partial
import math
import settings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from bn_lib.utils.pooling import WildcatPool2d, ClassWisePool
from torch.nn.modules.batchnorm import _BatchNorm
class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CAE(nn.Module):
    '''

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(CAE, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        z = None
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(x_t, mu)      # b * n * k
                z = F.softmax(z, dim=2)     # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))  # normalization
                mu = torch.bmm(x, z_)       # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        x = x.view(b, c, h, w)              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu, z_t.view(b, z_t.size(1), h, w)

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.\

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class CFEANetV3(nn.Module):
    def __init__(self, model,n_classes):
        super(CFEANetV3,self).__init__()
        self.features = nn.Sequential(
            # model.features,
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            # model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        for name, param in self.features.named_parameters():
            print(name)
            # if ("0.conv1.weight" == name) or ("0.bn1.weight" == name) or ("0.bn1.bias" == name) or ("1.0." in name) or  ("1.1." in name) or  ("1.2." in name):
            #     param.requires_grad=False
            if ("0.weight" == name) or ("1.weight" == name) or ("1.bias" == name):
                param.requires_grad=False
        # # self.features=model.features
        self.n_classes = n_classes
        self.fc0 = ConvBNReLU(512*4, 512, 3, 1, 1, 1)
        self.emau = CAE(512, 2*self.n_classes, settings.STAGE_NUM)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_base = nn.Linear(512, self.n_classes)
        self.sigmoid = nn.Sigmoid()
        # self.enc_class_conv = ClassWisePool(10)
        # self.enc_WildcatPool2d = WildcatPool2d(kmax=0.3, kmin=None, alpha=0.7)
        self.att_map=None
        # Put the criterion inside the model to make GPU load balanced
        # self.crit = nn.MultiLabelSoftMarginLoss()

    def forward(self, img):
        x = self.features(img)
        x = self.fc0(x)
        
        # x, mu, z_t = self.emau(x)
        # att_map = z_t
        # self.att_map=att_map
        # # z_t = self.enc_class_conv(z_t)
        # # z_e = self.enc_WildcatPool2d(z_t)
        # z_e = self.gap(z_t)
        # z_e = z_e.view(z_e.size(0), self.n_classes,
        #                z_t.size(1)//self.n_classes)
        # z_e = z_e.mean(dim=2)
        # z_e = z_e.view(z_e.size(0), -1)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        pred = self.fc_base(x)
        return pred


    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': filter(lambda p: p.requires_grad, self.features.parameters()), 'lr': lr*lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]


if __name__ == '__main__':
    model = CFEANet(n_classes=20)
    model.eval()
    print(list(model.named_children()))
    image = torch.randn(1, 3, 512, 512)
    label = torch.zeros(1, 20).long()
    loss, mu = model(image, label)
    print(loss)
    
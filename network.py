import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import MobileNetV2


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class ONet(nn.Module):

    def __init__(self):

        super(ONet, self).__init__()
        self.mobilenet = MobileNetV2(68*2)
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(3, 32, 3, 1)),
        #     ('prelu1', nn.PReLU(32)),
        #     ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

        #     ('conv2', nn.Conv2d(32, 64, 3, 1)),
        #     ('prelu2', nn.PReLU(64)),
        #     ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

        #     ('conv3', nn.Conv2d(64, 64, 3, 1)),
        #     ('prelu3', nn.PReLU(64)),
        #     ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

        #     ('conv4', nn.Conv2d(64, 128, 2, 1)),
        #     ('prelu4', nn.PReLU(128)),

        #     ('flatten', Flatten()),
        #     ('conv5', nn.Linear(15488, 256)),
        #     ('drop5', nn.Dropout(0.25)),
        #     ('prelu5', nn.PReLU(256)),
        # ]))

        # self.conv6_3 = nn.Linear(256, 68*2)


    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = x.float()
        c = self.mobilenet(x)
        return c


if __name__ == "__main__":
    net = ONet()
    data = torch.randn(2, 3, 112, 112)
    net(data)
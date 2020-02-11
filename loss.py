import math
import torch
from torch.nn import SmoothL1Loss, MSELoss
from config import weights

from torch import nn


weights = torch.FloatTensor(weights).cuda()


class WingLoss(nn.Module):
    def __init__(self, omega=10., epsilon=2.):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred

        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        
        index_y1 = (delta_y < self.omega).nonzero()
        index_y2 = (delta_y >= self.omega).nonzero()

        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C

        loss1 = weights[index_y1[:, 0]] * loss1
        loss2 = weights[index_y2[:, 0]] * loss2

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def landmakr_loss(x1, x2):
    x = x1 - x2
    x = x.view(x.size(0), -1, 2)

    loss = 0.5 *torch.norm(x, dim=2)
    loss = weights * loss
    loss = torch.sum(loss) / (x.size(0) * x.size(1))
    
    return loss


if __name__ == "__main__":
    wing_loss = WingLoss()
    a = torch.randn(2, 68*2)
    b = torch.randn(2, 68*2)

    print(wing_loss(a, b))
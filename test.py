import torch
import torch
import torch.optim as optim

from dataloader import FaceLandmarksDataset
from network import ONet
from loss import landmakr_loss
from headpose import get_head_pose
from utils import show_landmarks

import matplotlib.pyplot as plt

eopoch = 160


train_loader = torch.utils.data.DataLoader(
        FaceLandmarksDataset(r"E:\papers\face\test1220\test_lv", True),
        batch_size=1, shuffle=False)


net = ONet().cuda()
net.load_state_dict(torch.load('./checkpoints/new_data_20.pkl'))


optimizer = torch.optim.SGD(net.parameters(), 0.05,
                                momentum=0.9,
                                weight_decay=5e-4)

net.eval()


def get_pose_torch(img, landmark):
    landmark = landmark.tolist()
    get_head_pose(landmark, img)

for idx, data in enumerate(train_loader, 0):
    image = data['image'].cuda()
    # landmarks = data['landmarks']

    output = net(image)
    real_img = data['image'].cuda()
    img = real_img[0].permute(1, 2, 0).cpu().detach().numpy()

    landmark = output[0].view(-1, 2).cpu().detach().numpy()
    print(idx)
    show_landmarks(img, landmark)
    # get_pose_torch(img, landmark)
    input()

optimizer.step()
print("loss: {}".format(sum(loss_stack) / len(loss_stack)))
torch.save(net.state_dict(), "./checkpoints/{}.pkl".format(i))
import torch
import torch
import torch.optim as optim

from dataloader import FaceLandmarksDataset
from network import ONet
from loss import landmakr_loss, WingLoss


eopoch = 180


train_loader = torch.utils.data.DataLoader(
        FaceLandmarksDataset(r"E:\papers\face\datasets\datasets\gather", False),
        batch_size=32, shuffle=True)

net = ONet().cuda()
net.load_state_dict(torch.load('./checkpoints/new_data_3.pkl'))


optimizer = torch.optim.Adam(net.parameters(), 0.001)
wing_loss = WingLoss()


for i in range(1, eopoch):
    loss_stack = []
    for idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        image = data['image'].cuda()
        landmarks = data['landmarks'].cuda().float()

        output = net(image)
        loss = wing_loss(landmarks, output)
        
        loss_stack.append(loss.item())
        loss.backward()
        if idx % 20 == 0:
            print(sum(loss_stack) / len(loss_stack))

        optimizer.step()
    print("zepoch: {}, loss: {}".format(i, sum(loss_stack) / len(loss_stack)))
    torch.save(net.state_dict(), "./checkpoints/new_data_{}.pkl".format(i))
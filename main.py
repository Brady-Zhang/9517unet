from unet import U_Net
from dataset import dataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np



criterion = nn.CrossEntropyLoss()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataloader = DataLoader(dataset('./train'), batch_size=2,shuffle=True)
test_dataloader = DataLoader(dataset('./val'), batch_size=2,shuffle=True)

net=U_Net(img_ch=3,output_ch=18).to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)

deep_supervision = False

epochs=100


def IOU(pred, target, nclass=18):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for i in range(nclass):
        pred_ins = pred == i
        target_ins = target == i
        inser = (pred_ins & target_ins).sum().float()
        union = pred_ins.sum().float() + target_ins.sum().float() - inser
        if union == 0:
            iou = float('nan')
        else:
            iou = inser / (union + 1e-10)
        ious.append(iou)
    return torch.tensor(ious).mean().item()

def validate(val_dl, net):
    net.eval()
    ious = 0
    with torch.no_grad():
        for input, target in tqdm(val_dl):
            input = input.to(device)
            target = target.to(device)

            output = net(input)
            pred = torch.argmax(output, dim=1)
            ious += IOU(pred, target)

    iou = ious / len(val_dl)
    return iou

for i in range(epochs):
    net.train()
    mean_loss = []
    for data, label in tqdm(train_dataloader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label)
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    print('EPOCH:', i)
    print("LOSS:", sum(mean_loss) / len(mean_loss))

    log = validate(test_dataloader, net)
    print('IOU:', log)

# def IOU(pred, target, nclass=18):
#     ious = []
#     for i in range(nclass):
#         pred_ins = pred == i
#         target_ins = target == i
#         inser = pred_ins[target_ins].sum()
#         union = pred_ins.sum() + target_ins.sum() - inser
#         iou = inser / (union+1e-10)
#         ious.append(iou)

#     return sum(ious)/nclass



# def validate(val_dl, net):
#     # switch to evaluate mode
#     net.eval()
#     total = len(val_dl)
#     ious = 0
#     with torch.no_grad():
#         for input, target in tqdm(val_dl):
#             # print('input,target',input.shape)
#             # input = input.cuda()
#             # target = target.cuda()
#             input = data.to(torch.device('cpu'))
#             target = target.to(torch.device('cpu'))

#             output = net(input)
#             pred=torch.argmax(output,dim=1)
#             ious = ious + IOU(pred, target)

#     iou = ious / len(val_dl)
#     return iou

# for i in range(epochs):
#     mean_loss = []
#     for data,label in tqdm(train_dataloader):
#         data=data.to(device)
#         label = label.to(device)
#         optimizer.zero_grad()
#         output = net(data)
#         loss = criterion(output, label)
#         mean_loss.append(loss.item())
#         loss.backward()
#         optimizer.step()

#     print('EPOCH:',i)
#     print("LOSS:",sum(mean_loss) / len(mean_loss))

#     log=validate(test_dataloader, net)
#     print('IOU:',log)




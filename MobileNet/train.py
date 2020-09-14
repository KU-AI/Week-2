# Training code for the classification by ResNet50
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from mobilenet import MobileNetV2

import matplotlib.pyplot as plt

# Environment intiailization
class config():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.learning_rate = 1e-4
        self.epoch = 100
        self.batch_size = 32
        self.device = torch.device("cuda")
        self.save_path = "./weights/mobilenetv2_result.pth"
        self.image_list = "./data//pre_data/"
        a = os.listdir(self.image_list+"train")
        self.num_class = len(a)
        print("{} classes".format(self.num_class))


# main
if __name__ == '__main__':
    cfg = config()

    # Make dataloader_train&val
    train_dataset = datasets.ImageFolder(root=cfg.image_list+"train", transform=cfg.transform)
    val_dataset = datasets.ImageFolder(root=cfg.image_list+"val", transform=cfg.transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Declare models-resnet50
    model = MobileNetV2(num_classes = cfg.num_class)
    model.to(cfg.device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train_loss=[]
    val_loss=[]

    # Train by epoch-num
    for epoch in range(cfg.epoch):
        running_loss = 0.0
        model.train(True)
        for idx, data in enumerate(train_loader):
            img, target = data
            target = target.long()
            img, target = img.to(cfg.device), target.to(cfg.device)
            
            optimizer.zero_grad()

            outputs = model(img)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss.append(loss.item())
            if idx%10 == 9:
                print('train loss: [%d, %5d] loss: %.3f'%(epoch+1, cfg.batch_size*(idx+1), running_loss/10))
                running_loss = 0.0
        torch.save(model.state_dict(), cfg.save_path)

        running_loss = 0.0
        model.train(False) # model.eval()
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                img, target = data
                target = target.long()
                img, target = img.to(cfg.device), target.to(cfg.device)

                outputs = model(img)

                loss = criterion(outputs, target)
                running_loss += loss.item()
                val_loss.append(loss.item())
                if idx%10 == 9:
                    print('val loss: [%d, %5d] loss: %.3f'%(epoch+1, cfg.batch_size*(idx+1), running_loss/10))
                    running_loss = 0.0
    
    # draw a plot
    fig, loss_ax = plt.subplots()
    # acc_ax = loss_ax.twinx()
    loss_ax.plot(train_loss, 'y', label='train loss')
    loss_ax.plot(val_loss, 'r', label='val loss')
    # acc_ax.plot(##train_acc##, 'b', label='train acc')
    # acc_ax.plot(##val_acc##, 'g', label='val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    # acc_ax.set_ylabel('accuray')
    loss_ax.legend(loc='upper left')
    # acc_ax.legend(loc='lower left')
    plt.savefig("mobilenetv2.png")
    plt.show()

    print('Finished Training')


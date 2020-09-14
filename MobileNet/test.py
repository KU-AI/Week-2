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
            transforms.Resize((217,217)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.batch_size = 32
        self.device = torch.device("cuda")
        self.save_path = "./weights/mobilenetv2_result_weather_50.pth"
        self.image_list = "./data/pre_data/"
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
    model.load_state_dict(torch.load(cfg.save_path))
    model.to(cfg.device)
    model.eval()

    # Test
    count_right_train = 0
    count_wrong_train = 0
    for idx, data in enumerate(train_loader):
        img, target = data
        target = target.long()
        img, target = img.to(cfg.device), target.to(cfg.device)

        output = model(img)
        output = nn.Sequential(nn.Softmax())(output)

        for idx, vec in enumerate(output, 0):
            vec = list(vec)
            j = vec.index(max(vec))
            k = target[idx]
            # print('j: %.2i, k: %.2i' % (j, k))
            if j == k:
                count_right_train += 1
            else:
                count_wrong_train += 1

    count_right_val = 0
    count_wrong_val = 0
    for idx, data in enumerate(val_loader):
        img, target = data
        target = target.long()
        img, target = img.to(cfg.device), target.to(cfg.device)

        output = model(img)
        output = nn.Sequential(nn.Softmax())(output)

        for idx, vec in enumerate(output, 0):
            vec = list(vec)
            j = vec.index(max(vec))
            k = target[idx]
            # print('j: %.2i, k: %.2i' % (j, k))
            if j == k:
                count_right_val += 1
            else:
                count_wrong_val += 1

    print('count_right: {}, count_wrong: {}'.format(count_right_train, count_wrong_train))
    accuracy_train = 100*count_right_train/(count_right_train+count_wrong_train)
    print('Train accuracy: {:.2f}'.format(accuracy_train))

    print('count_right: {}, count_wrong: {}'.format(count_right_val, count_wrong_val))
    accuracy_val = 100*count_right_val/(count_right_val+count_wrong_val)
    print('Val accuracy: {:.2f}'.format(accuracy_val))

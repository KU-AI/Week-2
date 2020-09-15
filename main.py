import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
import torchvision.models as  models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
# from train import train_model, test
# from utils import encode_labels, plot_history
import matplotlib.pyplot as plt
import os
import torch.utils.model_zoo as model_zoo
from resnet50 import Resnet50
from train import train


data_dir = "weather"
device = torch.device("cuda:0")
# classes = ('n0, )
epoch=50
mean = [0.5, 0.5, 0.5]
std = [0.3, 0.3, 0.3]

model=Resnet50()
model.to(device)

# print(model)
loss_fc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

trans = transforms.Compose([ transforms.Resize((227,227)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
])


train_data = ImageFolder(root=os.path.join(data_dir,'train'), transform= trans)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)


val_data = ImageFolder(root=os.path.join(data_dir,'validation'), transform=trans)

val_loader=DataLoader(val_data, batch_size=4, shuffle=False, num_workers=2)


# tr_loss, val_loss=train(model=model, device=device, train_loader=train_loader, val_loader=val_loader, epochs=epoch)



# xi = [i for i in range(0, len(tr_loss[0]))]
# plt.plot(tr_loss[0], label='train')
# plt.plot(val_loss[0],label = 'val')
# plt.xticks(xi)
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


###############test accuracy########3333
count_right = 0
count_wrong = 0

for idx, data in enumerate(val_loader):
    image, target = data
    target = target.long()
    image, target = image.to(device), target.to(device)

    output=model(image)
    model.load_state_dict(torch.load('./best_weight/best_weather_weight.pth'))
    model.eval()
    output = nn.Sequential(nn.Softmax())(output)

    for idx,vec in enumerate(output):
        vec =list(vec)
        j=vec.index(max(vec))
        k=target[idx]

        print('j : %.2i, k : %.2i' %(j, k))
        if j==k:
            count_right+=1
        else:
            count_wrong+=1

print('count_right : {}, count_worong: {}'.format(count_right, count_wrong))
accuracy = 100*count_right/(count_right+count_wrong)
print(accuracy)

    












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
import os
import torch.utils.model_zoo as model_zoo
from resnet50 import Resnet50



def train(model, device, train_loader, val_loader, epochs):
    loss_fc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    val_max=100.0
    tr_loss = []
    val_loss = []

    for epoch in range(epochs):
        runing_loss=0.0
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)

                for i, data  in enumerate(train_loader):
                    image, target = data
                    image, target = image.to(device), target.to(device)
                    img= Variable(image) ## input image
            
                    optimizer.zero_grad()
                    output=model(img)

                    
                    loss = loss_fc(output,target)
                    loss.backward()
                    optimizer.step()
                    
                    
                    num_samples=float(len(train_loader))
            
                    runing_loss+=loss.item()
                    
                    
                    if i % num_samples == num_samples-1 :
                        print('[%d, %5d] train loss %.3f' % (epoch +1, i+1, runing_loss/ num_samples))
                        runing_loss=0.0

                tr_loss_ =runing_loss/num_samples
                tr_loss.append(tr_loss_)          
            
                   
            else:
                model.train(False) 

                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        image, target = image.to(device), target.to(device)
                        img = Variable(image)
                        
                        output = model(img)
                        loss = loss_fc(output, target)
                        
                        num_sample = float(len(val_loader))
                
                        runing_loss+=loss.item()

                        if i % num_sample ==  num_sample-1:
                            print('[%d, %5d] validation loss %.3f' % (epoch +1, i+1, runing_loss/ num_sample))
                            runing_loss=0.0
                            if runing_loss/num_sample <= val_max:
                                val_max = runing_loss/num_sample 
                                torch.save(model.state_dict(), 'best_weight/best_weather_weight.pth')

                    val_loss_ = runing_loss/num_sample
                    val_loss.append(val_loss_)     
    return ([tr_loss], [val_loss])                
                        
                                
                                  
                            

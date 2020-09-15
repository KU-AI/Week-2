import torch
import numpy as np
from torchvision.models.resnet import resnet50
from resnet50 import Resnet50

if __name__ == "__main__":

    model1 = resnet50(pretrained=False, progress=False)
    model1.to(torch.device("cuda:0"))

    with open("./resnet50_torch.txt","w") as f:
        text = str(model1)
        f.write(text)

    model2 = Resnet50()
    model2.to(torch.device("cuda:0"))
    
    with open("./resnet50_yours.txt","w") as f:
        text = str(model2)
        f.write(text)

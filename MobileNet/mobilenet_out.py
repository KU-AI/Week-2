from torchvision.models.mobilenet import mobilenet_v2
from mobilenet import MobileNetV2
import torch
import numpy as np


if __name__ == '__main__':

    model1 = mobilenet_v2(pretrained=False, progress=False)
    model1.to(torch.device("cuda:0"))
    with open("./mobilenet.txt", "w") as f:
        text = str(model1)
        f.write(text)
    model1.eval()

    model2 = MobileNetV2(num_classes=1000)
    model2.to(torch.device("cuda:0"))
    with open("./mobilenet2.txt", "w") as f:
        text = str(model2)
        f.write(text)
    model2.eval()
    
    k = np.zeros([1,3,224,224])
    k = torch.from_numpy(k).float()
    k = k.to(torch.device("cuda:0"))

    output1 = model1(k)
    print(output1.shape)
    output2 = model2(k)
    print(output2.shape)

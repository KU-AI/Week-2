import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url

def conv3x3(in_channel, out_channel, stride=1, dilation=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride = stride, padding = 1, 
                    bias=False)

def conv1x1(in_channel, out_channel, stride =1):
    return nn.Conv2d(in_channel, out_channel ,1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channel ,plane, dawnsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1=conv1x1(in_channel, plane)
        self.b1 = nn.BatchNorm2d(plane)
        self.conv2=conv3x3(plane, plane, stride)
        self.b2 = nn.BatchNorm2d(plane)
        self.conv3=conv1x1(plane, plane*4)
        self.b3 = nn.BatchNorm2d(plane*4)
        self.relu = nn.ReLU(inplace=True)
        self.dawnsample=dawnsample

    def forward(self, x):
        identity=x
        
        out = self.conv1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.b3(out)

        if self.dawnsample is not None:
            identity = self.dawnsample(x)

        out += identity
        out = self.relu(out)
        
        return out
## layer변경으로 다른 resnet 모델 사용 가능
class Resnet50(nn.Module):
    def __init__(self, layers=[3,4,6,3], num_classes=5):
        super(Resnet50, self).__init__()
        self.inplanes=64
        

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, layers[0], stride=1)

        self.layer2 = self.make_layer(128, layers[1], stride=2)

        self.layer3 = self.make_layer(256, layers[2], stride=2)

        self.layer4 = self.make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def make_layer(self, plane, layer, stride=1):
            
        dawnsample = None
        if stride != 1 or self.inplanes != plane * 4:
            dawnsample = nn.Sequential(
                conv1x1(self.inplanes, plane * 4, stride),
                nn.BatchNorm2d(plane * 4),
            )
        
        layers = []
        layers.append(Bottleneck(self.inplanes, plane, dawnsample=dawnsample, stride=stride))
        self.inplanes = plane * 4
        for i in range(1, layer):
            layers.append(Bottleneck(self.inplanes, plane))

        return nn.Sequential(*layers)
        

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return(x)



if __name__ == "__main__":
   model = Resnet50()
   print(model)


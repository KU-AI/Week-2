import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        padding = (kernel_size-1)//2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1): # in_channnels, c, t, s 
        super().__init__()
        self.use_res_connect = stride==1 and in_channels == out_channels
        hidden_dim = in_channels * expansion_factor
        layers = []
        if expansion_factor != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x+self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes = None):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        features = []
        # t,c,n,s configurations of Inverted residual bottleneck blocks
        # t: expansion factor, c: out_channels, n: repeated times, s: stride
        self.block_config = [[1,16,1,1],[6,24,2,2],[6,32,3,2],[6,64,4,2],
                             [6,96,3,1],[6,160,3,2],[6,320,1,1]]

        # conv layer on first
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True))
        
        # inverted residual bottleneck layers
        for t,c,n,s in self.block_config:
            for cnt in range(n):
                if cnt==0:
                    stride = s
                else:
                    stride = 1
                features.append(InvertedResidual(input_channel, c, t, stride))
                input_channel = c
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # denselayer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

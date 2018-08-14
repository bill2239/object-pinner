import torch.nn as nn
import torch

#lichi's modified net from Alexnet for finding phone  
class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        # output of this module as 27*27 heat map for phone location
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.feature_normal =nn.Sequential(
            nn.Conv2d(192,1,kernel_size=3, padding=1 )
        )
    #forward heat map as one hot vector
    def forward_map(self, x):
        x = self.features1(x)
        
        x=self.feature_normal(x)
        
        
        return x.view(x.size(0), 27 * 27)

    def forward(self, x):
        x = self.features1(x)
        
        x= self.features2(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
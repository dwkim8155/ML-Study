import torch
import torch.nn as nn
import torch.nn.functional as F

#Residual Block 구조 정의
class BasicBlock(nn.Module): #ResNet18, ResNet34
    mul = 1 #기본 구조에서는 out_chs의 수가 바뀌지 않음으로 1로 설정
    def __init__(self, in_chs:int, out_chs:int, stride=1):
        super().__init__()

        # 입력과 출력의 높이와 너비가 동일하고 identity mapping과의 연산을 위해 채널 수도 동일하게 조정 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size= 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chs)
        )

        #identity map
        self.shortcut = nn.Sequential()

        # F(x)와 x의 size가 안 맞을 경우, x의 모양을 맞춰 줌
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=1, stride = stride, bias=False),
                nn.BatchNorm2d(out_chs)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):#ResNet layer 50 이상
    mul = 4 
    def __init__(self, in_chs:int, out_chs:int, stride=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False), #BatchNorm에 편향이 있음으로 bias= False
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs*BottleNeck.mul, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_chs*BottleNeck.mul)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or out_chs*BottleNeck.mul != in_chs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs*BottleNeck.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs*BottleNeck.mul)
            )


    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


## 모델 생성
class ResNet(nn.Module):
    def __init__(self, block: BasicBlock or BottleNeck , num_blocks:list, num_classes=10):
        super().__init__()
        
        # 첫 conv layer num_out_chs 설정  
        self.init_out_chs = 64

        # 일반적인 conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.init_out_chs, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.init_out_chs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # residual conv block
        self.conv2_x = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3_x = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4_x = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5_x = self.make_layer(block, 512, num_blocks[3], stride=2)

        # avg, fc layers
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.mul, num_classes)

    def make_layer(self, block, out_chs, blocks:int, stride):
        # 첫 layer만 크기를 줄이고, 그 다음은 모양을 유지
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for i in range(blocks):
            layers.append(block(self.init_out_chs, out_chs, strides[i]))
            self.init_out_chs = block.mul * out_chs
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1) #(m,512) or (m,2048)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

if __name__ == '__main__':
    model = ResNet50()
    x = torch.randn(3,3,16,16)
    output = model(x)
    print(output.size())
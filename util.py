import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from torchinfo import summary
from natsort import natsorted
transform = transforms.Compose([
    transforms.Resize((270, 480)), 
    transforms.ToTensor()
])
class PairedDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        # 파일 이름을 자연스러운 순서로 정렬
        self.data_images = natsorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
        self.label_images = natsorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

    def __len__(self):
        return len(self.data_images)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_images[idx])
        label_path = os.path.join(self.label_dir, self.label_images[idx])

        data_image = Image.open(data_path).convert("RGB")
        label_image = Image.open(label_path).convert("L")
        print(torch.unique(label_image))  # 출력: tensor([0., 255.]) 또는 tensor([0., 1.])
        if self.transform:
            data_image = self.transform(data_image)
            label_image = self.transform(label_image)
        label_image[label_image > 0.5] = 1
        label_image[label_image <= 0.5] = 0
        print(torch.unique(label_image)) 
        return data_image, label_image

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels  # Depthwise Convolution
        )
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1  # Pointwise Convolutiondataloader
        )
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)  # BatchNorm 뒤에 ReLU 추가
print("ok2")
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3))  # Global Average Pooling
        y = y.view(batch, channels)  # Flatten for Linear layers
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)  # Reshape back to (B, C, 1, 1)
        return x * y


#=================================================================
# %%
# Define the model

class OptimizedLaneNet(nn.Module):
    def __init__(self):
        super(OptimizedLaneNet, self).__init__()
        # Encoder 1
        self.enc1_conv1 = DepthwiseSeparableConv(3, 32)
        self.enc1_conv2 = DepthwiseSeparableConv(32, 32)
        self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 2
        self.enc2_conv1 = DepthwiseSeparableConv(32, 64)
        self.enc2_conv2 = DepthwiseSeparableConv(64, 64)
        self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Block 1
        self.res1 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 128),
            DepthwiseSeparableConv(128, 64)
        )

                # Residual Block 2
        self.res2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 128),
            DepthwiseSeparableConv(128, 64)
        )

        # Decoder 1
        self.dec1_conv1 = DepthwiseSeparableConv(64, 64)  # 기존 32 → 64
        self.dec1_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1_conv2 = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 32)
        )

        # Decoder 2
        self.dec2_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2_conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # SE Block
        self.se = SEBlock(64,reduction=8)

    def forward(self, x):
        # Encoder 1
        x = F.relu(self.enc1_conv1(x))
        x = F.relu(self.enc1_conv2(x))
        x = self.enc1_pool(x)

        # Encoder 2
        x = F.relu(self.enc2_conv1(x))
        x = F.relu(self.enc2_conv2(x))
        x = self.enc2_pool(x)

        # SE Block (Context-aware scaling)
        x = self.se(x)

        # Residual Blocks
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))

        # Decoder 1
        x = F.relu(self.dec1_conv1(x))
        x = self.dec1_upsample(x)
        x = F.relu(self.dec1_conv2(x))

        # Decoder 2
        x = F.relu(self.dec2_conv1(x))
        x = self.dec2_upsample(x)
        x = torch.sigmoid(self.dec2_conv2(x))
        x = F.interpolate(x, size=(270, 480), mode='bilinear', align_corners=False)  # 라벨 크기와 맞춤

        return x

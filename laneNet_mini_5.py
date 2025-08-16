# %%
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


print("ok")

#=================================================================
# %%
# 
# 
import wandb


# %%
# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Define custom dataset for paired data
class PairedDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data_images = sorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
        self.label_images = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

    def __len__(self):
        return len(self.data_images)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_images[idx])
        label_path = os.path.join(self.label_dir, self.label_images[idx])

        data_image = Image.open(data_path).convert("RGB")
        label_image = Image.open(label_path).convert("L")

        if self.transform:
            data_image = self.transform(data_image)
            label_image = self.transform(label_image)
            label_image[label_image > 0.5] = 1
            label_image[label_image <= 0.5] = 0
        return data_image, label_image



# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((270, 480)), 
    transforms.ToTensor()
])
print("ok1")

#=================================================================
# %%
# Define paths
data_train_dir = '/home/jwoon/Desktop/my_ai_project/content/LaneDetection/data/TrainData/Data'
label_train_dir = '/home/jwoon/Desktop/my_ai_project/content/LaneDetection/data/TrainLabel/Label'
data_valid_dir = '/home/jwoon/Desktop/my_ai_project/content/LaneDetection/data/ValidData/Data'
label_valid_dir = '/home/jwoon/Desktop/my_ai_project/content/LaneDetection/data/ValidLabel/Label'
data_test_dir = '/home/jwoon/Desktop/my_ai_project/content/LaneDetection/data/TestData/Data'
label_test_dir = '/home/jwoon/Desktop/my_ai_project/content/LaneDetection/data/TestLabel/Label'

# Load datasets
train_dataset = PairedDataset(data_train_dir, label_train_dir, transform=transform)
valid_dataset = PairedDataset(data_valid_dir, label_valid_dir, transform=transform)
test_dataset = PairedDataset(data_test_dir, label_test_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 샘플 이미지 출력
def show_images(train_loader, num_images=4):
    data_iter = iter(train_loader)
    data_batch, label_batch = next(data_iter)

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    for i in range(num_images):
        # 데이터 이미지
        data_img = data_batch[i].permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        axes[i, 0].imshow(data_img)
        axes[i, 0].set_title("Data Image")
        axes[i, 0].axis("off")

        # 라벨 이미지
        label_img = label_batch[i].squeeze(0).numpy()  # (1, H, W) → (H, W)
        axes[i, 1].imshow(label_img, cmap="gray")
        axes[i, 1].set_title("Label Image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# 이미지 시각화
show_images(train_loader, num_images=4)

print("ok2")
# %%
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


# class LaneNet(nn.Module):
#     def __init__(self):
#         super(LaneNet, self).__init__()
#         # Encoder1
#         self.enc1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.enc1_bn1 = nn.BatchNorm2d(64)
#         self.enc1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.enc1_bn2 = nn.BatchNorm2d(64)
#         self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         # Encoder2
#         self.enc2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
#         self.enc2_bn1 = nn.BatchNorm2d(128)
#         self.enc2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.enc2_bn2 = nn.BatchNorm2d(128)
#         self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         # Bottleneck
#         self.bottleneck_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bottleneck_bn1 = nn.BatchNorm2d(256)
#         self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bottleneck_bn2 = nn.BatchNorm2d(256)

#         # Decoder1
#         self.dec1_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.dec1_bn1 = nn.BatchNorm2d(128)
#         self.dec1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.dec1_bn2 = nn.BatchNorm2d(128)

#         # Decoder2
#         self.dec2_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.dec2_bn1 = nn.BatchNorm2d(64)
#         self.dec2_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.dec2_bn2 = nn.BatchNorm2d(64)

#         # Final prediction
#         self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         # Encoder1
#         x = F.relu(self.enc1_bn1(self.enc1_conv1(x)))
#         x = F.relu(self.enc1_bn2(self.enc1_conv2(x)))
#         enc1_out = self.enc1_pool(x)  # Skip connection from enc1_out

#         # Encoder2
#         x = F.relu(self.enc2_bn1(self.enc2_conv1(enc1_out)))
#         x = F.relu(self.enc2_bn2(self.enc2_conv2(x)))
#         enc2_out = self.enc2_pool(x)  # Skip connection from enc2_out

#         # Bottleneck
#         x = F.relu(self.bottleneck_bn1(self.bottleneck_conv1(enc2_out)))
#         x = F.relu(self.bottleneck_bn2(self.bottleneck_conv2(x)))

#         # Decoder1
#         x = F.relu(self.dec1_bn1(self.dec1_conv1(x)))
#         x = F.relu(self.dec1_bn2(self.dec1_conv2(x)))
#         x = F.interpolate(x, size=enc2_out.shape[2:], mode='bilinear', align_corners=False)  # Match size of enc2_out
#         x = x + enc2_out  # Skip connection

#         # Decoder2
#         x = F.relu(self.dec2_bn1(self.dec2_conv1(x)))
#         x = F.relu(self.dec2_bn2(self.dec2_conv2(x)))
#         x = F.interpolate(x, size=enc1_out.shape[2:], mode='bilinear', align_corners=False)  # Match size of enc1_out
#         x = x + enc1_out  # Skip connection

#         # Final prediction
#         x = F.interpolate(x, size=(270, 480), mode='bilinear', align_corners=False)  # Ensure final size is 512x512
#         x = torch.sigmoid(self.final_conv(x))  # Normalize output to [0, 1]
#         return x






print("ok3")
#=================================================================
# %%
# Initialize model, loss, and optimizer
model = OptimizedLaneNet()

summary(model, input_size=(1, 3, 270, 480))
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30

# Early stopping settings
best_valid_loss = float('inf')
early_stop_counter = 0
patience = 3


wandb.init(project="LaneNet", config={"epochs": num_epochs, "batch_size": 8, "lr": 0.001})
# %%
if torch.cuda.is_available():
    wandb.config.update({"gpu_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3)})


for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    
    # Log epoch-wise losses
    train_loss_avg = train_loss / len(train_loader)


    # Validation loop
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device).float(), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    valid_loss_avg = valid_loss / len(valid_loader)
    # Log losses to WandB
    train_loss_avg = train_loss / len(train_loader)
    valid_loss_avg = valid_loss / len(valid_loader)
    wandb.log({"Train Loss": train_loss_avg, "Validation Loss": valid_loss_avg, "Epoch": epoch + 1})

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {valid_loss/len(valid_loader):.4f}")
    # Early stopping
    if valid_loss_avg < best_valid_loss:
        best_valid_loss = valid_loss_avg
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model_240_1.pth')
        print("Validation loss improved. Model saved.")
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

print("ok4")
wandb.finish()

#=================================================================
# %%
# Save the trained model
# 모델 저장
torch.save(model, "lanenet_model_240.pt")

model = OptimizedLaneNet()
model.load_state_dict(torch.load('best_model_240_1.pth'))  # 사전 학습된 가중치 로드
model = model.to('cuda')  # GPU로 이동

# Test the model
model.eval()
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.float(), test_labels.float()
test_images = test_images.to(device)

# Generate predictions
with torch.no_grad():
    test_outputs = model(test_images)
    test_preds = (test_outputs > 0.5).float()


# %%
# Plot results
random_indices = random.sample(range(test_images.size(0)), 8)


for i, idx in enumerate(random_indices):
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(test_images[idx].permute(1, 2, 0).cpu().numpy())

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(test_preds[idx][0].cpu().numpy(), cmap="gray")

    # Ground Truth Mask
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")
    plt.imshow(test_labels[idx][0].cpu().numpy(), cmap="gray")

    plt.show()

# %%
import cv2
import numpy as np
from ptflops import get_model_complexity_info
input_size = (3, 270, 480)

# FLOPs와 파라미터 계산
flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")
# %%
# Test images
image_paths = ["test_image1.png", "test_image2.png", "test_image3.png", "test_image4.png","test_image5.png","test_image6.png","test_image7.png","test_image8.png","test_image9.png","test_image10.png"]  # Replace with your image paths

fig, axes = plt.subplots(10, 2, figsize=(20, 40))
transform = transforms.Compose([
    transforms.ToPILImage(),  # numpy 배열을 PIL 이미지로 변환
    transforms.Resize((270, 480)), 
    transforms.ToTensor()
])

for i, image_path in enumerate(image_paths):
    # Load and preprocess image
    image = cv2.imread(image_path)
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        output = output.squeeze().cpu().numpy()

    # Threshold the output to create a binary mask
    mask = (output > 0.2).astype(np.uint8) * 255

    # Display original and predicted images
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original Image {i + 1}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title(f"Predicted Mask {i + 1}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

# %%
for i in range(5):  # 5개만 확인
    data, label = train_dataset[i]  # 데이터셋에서 직접 접근
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(data.permute(1, 2, 0))  # (C, H, W) → (H, W, C)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label.squeeze(0), cmap='gray')  # 라벨은 단일 채널
    plt.title("Ground Truth")
    plt.axis("off")
    plt.show()
# %%

#%%###################################################################
from util import OptimizedLaneNet
#%%
import os
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted

#%%###################################################################

class PairedDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

        self.data_images = natsorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
        self.label_images = natsorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

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

transform = transforms.Compose([
    transforms.Resize((270, 480)), 
    transforms.ToTensor()          
])
#%%############33
data_train_dir = '/home/jwoon/Desktop/my_ai_project/augmented_data_/augmented_train_image'
label_train_dir = '/home/jwoon/Desktop/my_ai_project/augmented_data_/augmented_train_label'
data_valid_dir = '/home/jwoon/Desktop/my_ai_project/augmented_data_/augmented_vali_image'
label_valid_dir = '/home/jwoon/Desktop/my_ai_project/augmented_data_/augmented_vali_label'
data_test_dir = '/home/jwoon/Desktop/my_ai_project/augmented_data_/augmented_test_image'
label_test_dir = '/home/jwoon/Desktop/my_ai_project/augmented_data_/augmented_test_label'

# Load datasets
train_dataset = PairedDataset(data_train_dir, label_train_dir, transform=transform)
valid_dataset = PairedDataset(data_valid_dir, label_valid_dir, transform=transform)
test_dataset = PairedDataset(data_test_dir, label_test_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
data_iter = iter(train_loader)
images, masks = next(data_iter)


print(f"Type of images: {type(images)}") 
print(f"Type of masks: {type(masks)}")   
print(f"Unique values in masks: {torch.unique(masks)}") 
#%%###################################################################
model = OptimizedLaneNet()
model.load_state_dict(torch.load('best_model_240_1.pth'))
model = model.to('cuda')


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) 


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
#%%###################################################################

num_epochs = 50
early_stop_patience = 10
best_val_loss = np.inf
patience_counter = 0


for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to('cuda').float(), masks.to('cuda').float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to('cuda'), masks.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(valid_loader)


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_finetuned_model.pt')
        print("Validation loss improved. Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered. Training stopped.")
            break


    scheduler.step(val_loss)


# %%###############################################
model = OptimizedLaneNet()


model.load_state_dict(torch.load('best_finetuned_model.pt'))
model = model.to('cuda')
model.eval()

# Test Set 평가
test_loss = 0.0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to('cuda'), masks.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")


#%%###################################################################

import matplotlib.pyplot as plt
import random
#%%############333
# 랜덤 5개 샘플 선택
random_indices = random.sample(range(len(test_dataset)), 5)

# 시각화
fig, axes = plt.subplots(5, 3, figsize=(15, 25))  # (입력 이미지, 라벨, 예측) 3열 구성
for idx, random_idx in enumerate(random_indices):
    image, mask = test_dataset[random_idx]
    image_tensor = image.unsqueeze(0).to('cuda')  # 배치 차원 추가
    mask_tensor = mask.unsqueeze(0).to('cuda')

    # 모델 예측
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = (pred_mask > 0.5).float()  # Threshold 적용

    # Tensor → Numpy 변환
    image_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (채널, 높이, 너비) → (높이, 너비, 채널)
    mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()  # (1, 높이, 너비) → (높이, 너비)
    pred_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy()  # (1, 높이, 너비) → (높이, 너비)

    # 시각화
    axes[idx, 0].imshow(image_np)
    axes[idx, 0].set_title("Input Image")
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(mask_np, cmap='gray')
    axes[idx, 1].set_title("Ground Truth")
    axes[idx, 1].axis("off")

    axes[idx, 2].imshow(pred_np, cmap='gray')
    axes[idx, 2].set_title("Predicted Mask")
    axes[idx, 2].axis("off")

plt.tight_layout()
plt.show()

# %%
for i in range(10):  # 5개만 확인
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
import cv2
model = OptimizedLaneNet()

image_paths = ["test_image1.png", "test_image2.png", "test_image3.png", "test_image4.png","test_image5.png","test_image6.png","test_image7.png","test_image8.png","test_image9.png","test_image10.png"]  # Replace with your image paths
fig, axes = plt.subplots(10, 2, figsize=(20, 40))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('best_finetuned_model.pt'))
model = model.to(device)

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
from torchinfo import summary

summary(model, input_size=(1, 3, 270, 480))

# %%

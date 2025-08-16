#%%
from util import OptimizedLaneNet
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import numpy as np
import time
import glob

model = OptimizedLaneNet()
model.load_state_dict(torch.load('best_finetuned_model.pt'))

image_paths = sorted(glob.glob("test_image/test_image*.png"))
# Replace with your image paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
avg_time = 0.0

transform = transforms.Compose([
    transforms.ToPILImage(),  # numpy 배열을 PIL 이미지로 변환
    transforms.Resize((270, 480)), 
    transforms.ToTensor()
])
fig, axes = plt.subplots(10, 2, figsize=(20, 40))
avg_time = 0
for i, image_path in enumerate(image_paths):
    start = time.time()

    # Load and preprocess image
    image = cv2.imread(image_path)
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        output = output.squeeze().cpu().numpy()
        end = time.time()
        avg_time = avg_time + (end-start)


    # Threshold the output to create a binary mask
    mask = (output > 0.2).astype(np.uint8) * 255

    # Display original and predicted images
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original Image {i + 1}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title(f"Predicted Mask {i + 1}")
    axes[i, 1].axis("off")
print(avg_time / 10)
# %%

#%%
from util import OptimizedLaneNet
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import numpy as np
import time
model = OptimizedLaneNet()
model.load_state_dict(torch.load('best_finetuned_model.pt'))

image_paths = ["test_image1.png", "test_image2.png", "test_image3.png", "test_image4.png","test_image5.png","test_image6.png","test_image7.png","test_image8.png","test_image9.png","test_image10.png"]  # Replace with your image paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),  # numpy 배열을 PIL 이미지로 변환
    transforms.Resize((270, 480)), 
    transforms.ToTensor()
])
fig, axes = plt.subplots(10, 2, figsize=(20, 40))
avg_time = 0
for i, image_path in enumerate(image_paths):
    # Load and preprocess image
    image = cv2.imread(image_path)
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        start = time.time()
        output = model(input_image)
        end = time.time()
        avg_time = avg_time + (end-start)
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
print(avg_time / 10)
# %%

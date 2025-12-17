# CODE TO CHECK TEST IMAGES
# Required Libraries
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CBAM U-Net model definition (ensure class CBAMUNet is defined beforehand)
model = CBAMUNet(num_classes=5).to(device)
model.load_state_dict(torch.load(r"C:\Users\anoushka chatterjee\Desktop\u net\mars_unet_cbam_model.pth", map_location=device))
model.eval()

# Define transformation (adjust as per your training)
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

# Define color mapping for visualization
CLASS_COLOR = {
    0: (0, 0, 0),         # Background
    1: (255, 255, 0),     # Crater
    2: (255, 0, 0),       # Rough
    3: (0, 255, 0),       # Smooth
    4: (0, 0, 255),       # Alluvial_Fan
}

def decode_segmap(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, v in CLASS_COLOR.items():
        color_mask[mask == k] = v
    return color_mask

def load_image(path):
    return transform(Image.open(path).convert("RGB"))

# Path to test images
test_image_dir = r"D:\test images"
image_files = sorted(os.listdir(test_image_dir))

# Number of test images to visualize
n = 30

for i in range(n):
    img_path = os.path.join(test_image_dir, image_files[i])
    img_tensor = load_image(img_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        pred_mask_colored = decode_segmap(pred_mask)

    # Plot input image and predicted mask
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(Image.open(img_path))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask_colored)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

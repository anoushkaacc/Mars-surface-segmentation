# CODE FOR VISUALIZATION OF IMAGE - GT PAIR
# Required Libraries
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import torch.nn as nn

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color map for predicted masks
CLASS_COLORS = {
    0: (0, 0, 0),          # Background
    1: (255, 255, 0),      # Crater
    2: (255, 0, 0),        # Rough
    3: (0, 255, 0),        # Smooth
    4: (0, 0, 255)         # Alluvial_Fan
}

# CBAM Blocks and U-Net Definition
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg, max_], dim=1)))

class ConvBlockCBAM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlockCBAM(in_c, out_c)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        f = self.conv(x)
        p = self.pool(f)
        return f, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlockCBAM(out_c*2, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class CBAMUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.enc1 = EncoderBlock(3,  64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128,256)
        self.enc4 = EncoderBlock(256,512)
        self.bottleneck = ConvBlockCBAM(512,1024)
        self.dec1 = DecoderBlock(1024,512)
        self.dec2 = DecoderBlock(512,256)
        self.dec3 = DecoderBlock(256,128)
        self.dec4 = DecoderBlock(128,64)
        self.final = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        s1,p1 = self.enc1(x)
        s2,p2 = self.enc2(p1)
        s3,p3 = self.enc3(p2)
        s4,p4 = self.enc4(p3)
        b     = self.bottleneck(p4)
        d1    = self.dec1(b,   s4)
        d2    = self.dec2(d1,  s3)
        d3    = self.dec3(d2,  s2)
        d4    = self.dec4(d3,  s1)
        return self.final(d4)

# Load model state dict
model = CBAMUNet(num_classes=5).to(device)
model.load_state_dict(torch.load(r"D:\u net\mars_unet_cbam_model.pth", map_location=device))
model.eval()

# Transformation (must match training)
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

# Helper functions
def load_image(path):
    return transform(Image.open(path).convert("RGB"))

def load_mask(path):
    return np.array(Image.open(path).convert("RGB"))

def decode_segmap(mask):
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        output[mask == cls] = color
    return output

# Paths
image_dir = r"D:\new base\New folder\train\image"
mask_dir = r"D:\new base\New folder\train\mask"

image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

# Number of samples to visualize
n = 20

for i in range(n):
    img_path = os.path.join(image_dir, image_files[i])
    mask_path = os.path.join(mask_dir, mask_files[i])

    img_tensor = load_image(img_path).unsqueeze(0).to(device)
    gt_mask_img = Image.open(mask_path).convert("RGB")

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        pred_color_mask = decode_segmap(pred_mask)

    # Plot
    plt.figure(figsize=(6,2))

    plt.subplot(1, 3, 1)
    #plt.title("Input Image")
    plt.imshow(Image.open(img_path))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    #plt.title("Ground Truth")
    plt.imshow(gt_mask_img)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    #plt.title("Predicted Mask")
    plt.imshow(pred_color_mask)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

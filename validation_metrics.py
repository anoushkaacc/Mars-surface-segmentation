# ===========================
# CBAM-UNet Evaluation Cell (BACKGROUND IGNORED)
# ===========================
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os
import torch.nn as nn
from tqdm import tqdm

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Class Color Mapping
# ---------------------------
CLASS_COLORS = {
    0: (0, 0, 0),          # Background
    1: (255, 255, 0),      # Crater
    2: (255, 0, 0),        # Rough
    3: (0, 255, 0),        # Smooth
    4: (0, 0, 255)         # Alluvial_Fan
}
COLOR_TO_CLASS = {v: k for k, v in CLASS_COLORS.items()}

# ---------------------------
# CBAM-UNet Architecture
# ---------------------------
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
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

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
        return f, self.pool(f)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlockCBAM(out_c * 2, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class CBAMUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.bottleneck = ConvBlockCBAM(512, 1024)
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)
        return self.final(d4)

# ---------------------------
# Load Model
# ---------------------------
model = CBAMUNet(num_classes=5).to(device)
model.load_state_dict(torch.load(r"D:\u net\mars_unet_cbam_model.pth", map_location=device))
model.eval()

# ---------------------------
# Transforms
# ---------------------------
img_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

def load_rgb_mask_as_class(path):
    mask = Image.open(path).convert("RGB").resize((512, 512), Image.NEAREST)
    mask_np = np.array(mask)
    class_mask = np.zeros((512, 512), dtype=np.int64)
    for color, cls in COLOR_TO_CLASS.items():
        class_mask[np.all(mask_np == color, axis=-1)] = cls
    return class_mask

# ---------------------------
# Dataset paths
# ---------------------------
image_dir = r"D:\new base\base\originals\image"
mask_dir  = r"D:\new base\base\originals\mask"

image_files = sorted(os.listdir(image_dir))
mask_files  = sorted(os.listdir(mask_dir))
assert len(image_files) == len(mask_files), "Number of images and masks must match."

# ---------------------------
# Aggregated confusion statistics (to compute metrics excluding background)
# ---------------------------
num_classes = 5
TP_sum = np.zeros(num_classes, dtype=np.int64)
FP_sum = np.zeros(num_classes, dtype=np.int64)
FN_sum = np.zeros(num_classes, dtype=np.int64)
total_images = 0

with torch.no_grad():
    for img_f, mask_f in tqdm(zip(image_files, mask_files), total=len(image_files)):
        img = img_transform(Image.open(os.path.join(image_dir, img_f)).convert("RGB"))
        img = img.unsqueeze(0).to(device)
        gt  = load_rgb_mask_as_class(os.path.join(mask_dir, mask_f))
        pred = torch.argmax(model(img), dim=1).squeeze().cpu().numpy()

        # accumulate per-class TP, FP, FN
        for c in range(num_classes):
            pred_c = (pred == c)
            gt_c   = (gt == c)
            TP_sum[c] += np.logical_and(pred_c, gt_c).sum()
            FP_sum[c] += np.logical_and(pred_c, ~gt_c).sum()
            FN_sum[c] += np.logical_and(~pred_c, gt_c).sum()

        total_images += 1

# ---------------------------
# Compute metrics (exclude class 0)
# ---------------------------
eps = 1e-7
classes_to_eval = list(range(1, num_classes))  # ignore background class 0

IoU = []
Dice = []
Precision = []
Recall = []

for c in classes_to_eval:
    TP = TP_sum[c].astype(np.float64)
    FP = FP_sum[c].astype(np.float64)
    FN = FN_sum[c].astype(np.float64)

    iou  = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    prec = TP / (TP + FP + eps)
    rec  = TP / (TP + FN + eps)

    IoU.append(iou)
    Dice.append(dice)
    Precision.append(prec)
    Recall.append(rec)

IoU = np.array(IoU)
Dice = np.array(Dice)
Precision = np.array(Precision)
Recall = np.array(Recall)

# ---------------------------
# Results (per-class and macro mean excluding background)
# ---------------------------
print(f"Images evaluated: {total_images}\n")
for idx, c in enumerate(classes_to_eval, start=1):
    print(f"Class {c}: IoU={IoU[idx-1]:.4f}, Dice={Dice[idx-1]:.4f}, "
          f"Precision={Precision[idx-1]:.4f}, Recall={Recall[idx-1]:.4f}")

print("\nMacro Average (excluding background):")
print(f"IoU: {IoU.mean():.4f}")
print(f"Dice: {Dice.mean():.4f}")
print(f"Precision: {Precision.mean():.4f}")
print(f"Recall: {Recall.mean():.4f}")

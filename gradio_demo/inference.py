"""
Inference script for Watermark Injection Model

Usage:
    python inference.py
"""

import random
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================================
# Configuration - SET THESE VALUES
# ============================================================================

MODEL_PATH = "/teamspace/studios/this_studio/photomaker-spatial-multi-id-layered/gradio_demo/watermark-addition_watermark_model_watermark_injection_model.pth"  # Path to full model
INPUT_IMAGE = "/teamspace/studios/this_studio/photomaker-spatial-multi-id-layered/Data/Output/output_seed138148939_1.png"                             # Input image path
OUTPUT_IMAGE = "/teamspace/studios/this_studio/photomaker-spatial-multi-id-layered/Data/Output/output_seed138148939_1_W.png"                                # Output path (None = auto-generate)
CORNER = None                                        # 'top-left', 'top-right', 'bottom-left', 'bottom-right', or None for random


# ============================================================================
# Corner Configuration
# ============================================================================

CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right']


# ============================================================================
# Model Architecture (required for torch.load to unpickle)
# ============================================================================

class AlphaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = resnet.conv1.weight
            self.conv1.weight[:, 3:] = 0

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []
        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


class AlphaDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_up = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.output = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        f0, f1, f2, f3, f4 = features

        x = self.up4(f4)
        x = self._match_size(x, f3)
        x = torch.cat([x, f3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = self._match_size(x, f2)
        x = torch.cat([x, f2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = self._match_size(x, f1)
        x = torch.cat([x, f1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = self._match_size(x, f0)
        x = torch.cat([x, f0], dim=1)
        x = self.conv1(x)

        x = self.final_up(x)
        alpha = self.output(x)

        return alpha

    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x


class WatermarkInjectionNetwork(nn.Module):
    def __init__(self, templates: Dict[str, torch.Tensor],
                 masks: Dict[str, torch.Tensor],
                 max_alpha: float = 0.8):
        super().__init__()

        self.encoder = AlphaEncoder()
        self.decoder = AlphaDecoder()
        self.max_alpha = max_alpha

        for corner in CORNERS:
            self.register_buffer(f'template_{corner.replace("-", "_")}', templates[corner])
            self.register_buffer(f'mask_{corner.replace("-", "_")}', masks[corner])

    def get_template(self, corner: str) -> torch.Tensor:
        return getattr(self, f'template_{corner.replace("-", "_")}')

    def forward(self, image: torch.Tensor, corner_names: list) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = image.shape
        device = image.device

        corner_spatial = torch.zeros(B, 1, H, W, device=device)
        region = H // 4

        for b, corner in enumerate(corner_names):
            if corner == 'top-left':
                corner_spatial[b, :, :region*2, :region*2] = 1.0
            elif corner == 'top-right':
                corner_spatial[b, :, :region*2, -region*2:] = 1.0
            elif corner == 'bottom-left':
                corner_spatial[b, :, -region*2:, :region*2] = 1.0
            elif corner == 'bottom-right':
                corner_spatial[b, :, -region*2:, -region*2:] = 1.0

        x = torch.cat([image, corner_spatial], dim=1)
        features = self.encoder(x)
        alpha = self.decoder(features)

        if alpha.shape[2:] != image.shape[2:]:
            alpha = F.interpolate(alpha, size=image.shape[2:], mode='bilinear', align_corners=False)

        alpha = alpha * self.max_alpha

        watermarked = torch.zeros_like(image)

        for b, corner in enumerate(corner_names):
            template = self.get_template(corner).to(device)

            if template.shape[1:] != image.shape[2:]:
                template = F.interpolate(template.unsqueeze(0), size=image.shape[2:],
                                        mode='bilinear', align_corners=False).squeeze(0)

            a = alpha[b]
            watermarked[b] = image[b] * (1 - a) + template * a

        return watermarked, alpha


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(model_path, device):
    """Load full model from .pth file."""
    print(f"Loading model: {model_path}")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def add_watermark(image_path, model, device, corner):
    """Add watermark to a single image."""
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = img.shape[:2]

    # Get model's expected image size from template buffer
    template = model.template_bottom_right
    image_size = template.shape[1]

    # Preprocess
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # BGR to RGB, HWC to CHW
    img_rgb = img_normalized[:, :, ::-1].copy()
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        output, _ = model(img_tensor, [corner])

    # Post-process
    watermarked = output.squeeze(0).cpu().numpy()
    watermarked = np.transpose(watermarked, (1, 2, 0))
    watermarked = watermarked[:, :, ::-1]
    watermarked = np.clip(watermarked * 255, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    watermarked_original_size = cv2.resize(watermarked, (original_w, original_h))

    return watermarked_original_size


def main():
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load model
    model = load_model(MODEL_PATH, device)

    # Select corner (random if not specified)
    corner = CORNER if CORNER else random.choice(CORNERS)
    print(f"Using corner: {corner}")

    # Process image
    input_path = Path(INPUT_IMAGE)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Generate output path if not provided
    if OUTPUT_IMAGE:
        output_path = Path(OUTPUT_IMAGE)
    else:
        corner_suffix = corner.replace('-', '_')
        output_path = input_path.parent / f"{input_path.stem}_watermarked_{corner_suffix}{input_path.suffix}"

    print(f"\nProcessing: {input_path}")

    # Add watermark
    watermarked = add_watermark(input_path, model, device, corner)

    # Save result
    cv2.imwrite(str(output_path), watermarked)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()

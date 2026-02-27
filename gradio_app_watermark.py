#!/usr/bin/env python3
"""
PhotoMaker + Watermark Pipeline - Gradio App

This app combines the PhotoMaker image generation pipeline with
watermark injection to produce watermarked AI-generated images.
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import random
from PIL import Image
import cv2
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import gradio as gr

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces
from gradio_demo.style_template import styles
from gradio_demo.aspect_ratio_template import aspect_ratios
from insightface.utils import face_align
import PIL.Image


# ============================================================================
# Configuration
# ============================================================================

MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"
ASPECT_RATIO_LABELS = list(aspect_ratios)
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_LABELS[0]

# Watermark model paths - adjust as needed
WATERMARK_MODEL_PATH = os.environ.get(
    "WATERMARK_MODEL_PATH",
    "/teamspace/studios/this_studio/watermark-addition/output_injection_512/full_model.pth"
)

CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right']

# Global variables for models (loaded once)
pipe = None
face_detector = None
watermark_model = None
device = None


# ============================================================================
# Watermark Model Architecture (for loading saved model)
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
    def __init__(self, templates, masks, max_alpha=0.8):
        super().__init__()
        self.encoder = AlphaEncoder()
        self.decoder = AlphaDecoder()
        self.max_alpha = max_alpha
        for corner in CORNERS:
            self.register_buffer(f'template_{corner.replace("-", "_")}', templates[corner])
            self.register_buffer(f'mask_{corner.replace("-", "_")}', masks[corner])

    def get_template(self, corner):
        return getattr(self, f'template_{corner.replace("-", "_")}')

    def forward(self, image, corner_names):
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
                template = F.interpolate(
                    template.unsqueeze(0), size=image.shape[2:],
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            a = alpha[b]
            watermarked[b] = image[b] * (1 - a) + template * a
        return watermarked, alpha


# ============================================================================
# Utility Functions
# ============================================================================

def get_device():
    """Detect the best available device."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"


def apply_style(style_name, positive, negative=""):
    """Apply style template to prompt."""
    default_style = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default_style])
    return p.replace("{prompt}", positive), n + ' ' + negative


def compute_active_slots(prompt, trigger_words):
    """Compute which identity slots are active based on prompt."""
    active_slots = []
    for i, word in enumerate(trigger_words):
        if word in prompt:
            active_slots.append(i)
    return active_slots


# ============================================================================
# Model Loading
# ============================================================================

def load_all_models():
    """Load all models at startup."""
    global pipe, face_detector, watermark_model, device

    device = get_device()
    print(f"Loading models on {device}...")

    # Load PhotoMaker pipeline
    base_model_path = "SG161222/RealVisXL_V4.0"

    torch_dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if device == "mps":
        torch_dtype = torch.float16

    # Load T2I adapter
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        torch_dtype=torch_dtype,
        variant="fp16",
    ).to(device)

    # Load main pipeline
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_path,
        adapter=adapter,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    # Load PhotoMaker adapter
    photomaker_ckpt = hf_hub_download(
        repo_id="TencentARC/PhotoMaker-V2",
        filename="photomaker-v2.bin",
        repo_type="model",
    )

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_words=["img1", "img2"],
        pm_version="v2",
    )

    pipe.id_encoder.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Wrap UNet with identity slot injection
    from photomaker.identity_slot_unet import IdentitySlotUNet
    pipe.unet = IdentitySlotUNet(
        pipe.unet,
        down_strength=0.2,
        mid_strength=1.2,
        up_strength=1.5,
        temperature=0.35
    )

    pipe.to(device)
    pipe.enable_routing = False
    pipe.enable_slot_injection = True

    # Load face detector
    if device == "cuda":
        providers = ["CUDAExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    face_detector = FaceAnalysis2(
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    face_detector.prepare(
        ctx_id=ctx_id,
        det_size=(1024, 1024),
        det_thresh=0.25
    )

    # Load watermark model
    if os.path.exists(WATERMARK_MODEL_PATH):
        watermark_model = torch.load(WATERMARK_MODEL_PATH, map_location=device, weights_only=False)
        watermark_model.to(device)
        watermark_model.eval()
        print(f"Watermark model loaded from {WATERMARK_MODEL_PATH}")
    else:
        watermark_model = None
        print(f"Watermark model not found at {WATERMARK_MODEL_PATH}")

    print("All models loaded successfully!")


# ============================================================================
# Generation Functions
# ============================================================================

def add_watermark_to_image(pil_image, corner):
    """Add watermark to a PIL image."""
    global watermark_model, device

    if watermark_model is None:
        return pil_image

    # Convert PIL to numpy (RGB)
    img = np.array(pil_image)
    original_h, original_w = img.shape[:2]

    # Get model's expected size
    template = watermark_model.template_bottom_right
    image_size = template.shape[1]

    # Resize and normalize
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # HWC -> CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        output, _ = watermark_model(img_tensor, [corner])

    # Post-process
    watermarked = output.squeeze(0).cpu().numpy()
    watermarked = np.transpose(watermarked, (1, 2, 0))
    watermarked = np.clip(watermarked * 255, 0, 255).astype(np.uint8)

    # Resize back
    watermarked_original = cv2.resize(watermarked, (original_w, original_h))

    return Image.fromarray(watermarked_original)


def generate_image(
    upload_images,
    prompt,
    negative_prompt,
    aspect_ratio_name,
    style_name,
    num_steps,
    style_strength_ratio,
    guidance_scale,
    seed,
    randomize_seed,
    enable_watermark,
    watermark_corner,
    progress=gr.Progress(track_tqdm=True)
):
    """Main generation function."""
    global pipe, face_detector, device

    # Validate inputs
    if upload_images is None or len(upload_images) == 0:
        raise gr.Error("Please upload at least one face image!")

    if not prompt:
        raise gr.Error("Please enter a prompt!")

    # Check for trigger words
    if "img1" not in prompt and "img2" not in prompt:
        gr.Warning("Prompt should contain 'img1' or 'img2' trigger words for best results")

    # Randomize seed if needed
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Get output dimensions
    output_w, output_h = aspect_ratios[aspect_ratio_name]
    print(f"[Debug] Generate image using aspect ratio [{aspect_ratio_name}] => {output_w} x {output_h}")

    # Apply style
    styled_prompt, styled_negative = apply_style(style_name, prompt, negative_prompt)
    print(f"[Debug] Prompt: {styled_prompt}")
    print(f"[Debug] Neg Prompt: {styled_negative}")

    # Load input images
    original_images = []
    for img_path in upload_images:
        original_images.append(load_image(img_path))

    # Extract face embeddings
    id_embed_list = []
    id_pixel_list = []
    input_id_images = []
    bbox_list = []

    for img_idx, img in enumerate(original_images):
        img_array = np.array(img)[:, :, ::-1]  # RGB -> BGR
        faces = analyze_faces(face_detector, img_array)

        if len(faces) == 0:
            gr.Warning(f"No face detected in image {img_idx + 1}")
            continue

        faces = sorted(faces, key=lambda f: f.bbox[0])

        for face_idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            W, H = img.width, img.height

            normalized_bbox = [x1/W, y1/H, x2/W, y2/H]
            bbox_list.append(normalized_bbox)

            id_embed_list.append(torch.from_numpy(face["embedding"]).float())

            aligned_face = face_align.norm_crop(
                img_array, landmark=face.kps, image_size=224
            )
            aligned_face = aligned_face[:, :, ::-1]
            aligned_face = PIL.Image.fromarray(aligned_face)

            pixel_tensor = pipe.image_processor.preprocess(aligned_face)[0]
            id_pixel_list.append(pixel_tensor)
            input_id_images.append(aligned_face)

    if len(id_embed_list) == 0:
        raise gr.Error("No valid faces found in any input images.")

    # Stack embeddings
    all_embeds = torch.stack(id_embed_list)
    all_embeds = F.normalize(all_embeds, dim=1)

    # Group by identity
    groups = []
    threshold = 0.5

    for idx, embed in enumerate(all_embeds):
        placed = False
        for group in groups:
            similarity = torch.dot(embed, all_embeds[group[0]]).item()
            if similarity > threshold:
                group.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])

    # Average embeddings per identity
    final_embeds = []
    final_pixels = []
    final_bboxes = []

    for group in groups:
        group_embeds = torch.stack([all_embeds[i] for i in group])
        avg_embed = F.normalize(group_embeds.mean(dim=0), dim=0)
        final_embeds.append(avg_embed)
        final_pixels.append(id_pixel_list[group[0]])
        final_bboxes.append(bbox_list[group[0]])

    id_embeds = torch.stack(final_embeds)
    id_pixel_values = torch.stack(final_pixels)
    identity_bboxes = torch.tensor(final_bboxes).float()

    # Sort left to right
    x_centers = [(bbox[0] + bbox[2]) / 2.0 for bbox in identity_bboxes]
    sorted_indices = sorted(range(len(x_centers)), key=lambda i: x_centers[i])

    identity_bboxes = identity_bboxes[sorted_indices]
    id_embeds = id_embeds[sorted_indices]
    id_pixel_values = id_pixel_values[sorted_indices]

    # Validate triggers
    trigger_words = ["img1", "img2"]
    active_slots = compute_active_slots(prompt, trigger_words)

    if len(active_slots) == 0:
        gr.Warning("No identity triggers (img1, img2) found in prompt. Using all detected identities.")
        active_slots = list(range(min(len(id_embeds), 2)))

    pipe.unet.set_active_slots(active_slots)

    # Add batch dimension
    id_embeds = id_embeds.unsqueeze(0)
    id_pixel_values = id_pixel_values.unsqueeze(0)

    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)

    # Move to device
    id_embeds = id_embeds.to(device=pipe.unet.device, dtype=pipe.unet.dtype)
    identity_bboxes = identity_bboxes.to(device=pipe.unet.device, dtype=pipe.unet.dtype)

    pipe.unet.set_identity_data(id_embeds, identity_bboxes)

    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    start_merge_step = min(start_merge_step, 30)

    # Generate
    print("Starting inference...")
    images = pipe(
        prompt=styled_prompt,
        width=output_w,
        height=output_h,
        negative_prompt=styled_negative,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        input_id_images=input_id_images,
        id_embeds=id_embeds,
        id_pixel_values=id_pixel_values,
        image=None,
        adapter_conditioning_scale=0.,
        adapter_conditioning_factor=0.,
        identity_bboxes=identity_bboxes,
    ).images

    pipe.unet.clear_identity_data()

    generated_image = images[0]

    # Apply watermark if enabled
    if enable_watermark and watermark_model is not None:
        corner = watermark_corner if watermark_corner != "random" else random.choice(CORNERS)
        watermarked_image = add_watermark_to_image(generated_image, corner)
        info_text = f"Seed: {seed} | Watermark: {corner}"
    else:
        watermarked_image = generated_image
        info_text = f"Seed: {seed} | Watermark: disabled"

    return generated_image, watermarked_image, info_text


# ============================================================================
# Gradio UI
# ============================================================================

def create_demo():
    """Create the Gradio demo interface."""

    title = """
    <h1 align="center">PhotoMaker + Watermark Pipeline</h1>
    <p align="center">Generate personalized images using PhotoMaker V2 and automatically add invisible watermarks.</p>
    """

    description = """
    **Steps:**
    1. Upload face image(s) - one image is ok, but more is better
    2. Enter a prompt with `img1` or `img2` trigger words (e.g., "a man img1 wearing a suit")
    3. Adjust settings and click Submit
    4. Get both the original and watermarked output!
    """

    with gr.Blocks() as demo:
        gr.HTML(title)
        gr.Markdown(description)

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                files = gr.Files(
                    label="Upload face image(s)",
                    file_types=["image"],
                    file_count="multiple",
                    height=200,
                )
                uploaded_gallery = gr.Gallery(
                    label="Uploaded Images Preview",
                    visible=True,
                    columns=4,
                    rows=2,
                    height=500,
                    object_fit="contain",
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a man img1 wearing a business suit, professional photo",
                    info="Use 'img1' and 'img2' as trigger words for identities",
                    lines=2
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry",
                    lines=2
                )

                with gr.Row():
                    style = gr.Dropdown(
                        label="Style",
                        choices=STYLE_NAMES,
                        value=DEFAULT_STYLE_NAME
                    )
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=ASPECT_RATIO_LABELS,
                        value=DEFAULT_ASPECT_RATIO
                    )

                with gr.Accordion("Advanced Options", open=False):
                    num_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=50,
                    )
                    style_strength_ratio = gr.Slider(
                        label="Style Strength (%)",
                        minimum=15,
                        maximum=50,
                        step=1,
                        value=20,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=5.0,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=True
                    )

                with gr.Accordion("Watermark Settings", open=True):
                    enable_watermark = gr.Checkbox(
                        label="Enable Watermark",
                        value=True
                    )
                    watermark_corner = gr.Dropdown(
                        label="Watermark Corner",
                        choices=["random"] + CORNERS,
                        value="random"
                    )

                submit_btn = gr.Button("Submit", variant="primary")

            # Right column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### Output")

                with gr.Row():
                    output_original = gr.Image(
                        label="PhotoMaker Output",
                        type="pil",
                        height=400,
                    )
                    output_watermarked = gr.Image(
                        label="Watermarked Output",
                        type="pil",
                        height=400,
                    )

                output_info = gr.Textbox(
                    label="Generation Info",
                    interactive=False
                )

        # Update gallery when files are uploaded
        def update_gallery(files):
            if files is None:
                return []
            return [f for f in files]

        files.change(
            fn=update_gallery,
            inputs=files,
            outputs=uploaded_gallery
        )

        # Submit button action
        submit_btn.click(
            fn=generate_image,
            inputs=[
                files,
                prompt,
                negative_prompt,
                aspect_ratio,
                style,
                num_steps,
                style_strength_ratio,
                guidance_scale,
                seed,
                randomize_seed,
                enable_watermark,
                watermark_corner,
            ],
            outputs=[output_original, output_watermarked, output_info]
        )

    return demo


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Load models at startup
    load_all_models()

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL that bypasses Lightning AI proxy issues
    )

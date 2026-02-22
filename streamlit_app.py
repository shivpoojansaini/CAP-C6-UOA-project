#!/usr/bin/env python3
"""
PhotoMaker + Watermark Pipeline - Streamlit App

This app combines the PhotoMaker image generation pipeline with
watermark injection to produce watermarked AI-generated images.
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import numpy as np
import random
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

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
# Model Loading (Cached)
# ============================================================================

@st.cache_resource
def load_photomaker_pipeline():
    """Load and cache the PhotoMaker pipeline."""
    device = get_device()
    st.info(f"Loading PhotoMaker pipeline on {device}...")

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

    return pipe, device


@st.cache_resource
def load_face_detector():
    """Load and cache the face detector."""
    device = get_device()

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

    return face_detector


@st.cache_resource
def load_watermark_model():
    """Load and cache the watermark injection model."""
    device = get_device()

    if not os.path.exists(WATERMARK_MODEL_PATH):
        st.warning(f"Watermark model not found at {WATERMARK_MODEL_PATH}")
        return None, device

    model = torch.load(WATERMARK_MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    return model, device


# ============================================================================
# PhotoMaker Generation
# ============================================================================

def generate_photomaker_image(
    pipe,
    face_detector,
    device,
    input_images,
    prompt,
    negative_prompt,
    style_name,
    output_width,
    output_height,
    num_steps,
    guidance_scale,
    style_strength_ratio,
    seed,
    num_outputs=1
):
    """Generate images using the PhotoMaker pipeline."""

    # Apply style
    styled_prompt, styled_negative = apply_style(style_name, prompt, negative_prompt)

    # Load input images
    original_images = []
    for img in input_images:
        if isinstance(img, str):
            original_images.append(load_image(img))
        else:
            original_images.append(img)

    # Extract face embeddings
    id_embed_list = []
    id_pixel_list = []
    input_id_images = []
    bbox_list = []

    for img_idx, img in enumerate(original_images):
        img_array = np.array(img)[:, :, ::-1]  # RGB -> BGR
        faces = analyze_faces(face_detector, img_array)

        if len(faces) == 0:
            st.warning(f"No face detected in image {img_idx + 1}")
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
        raise ValueError("No valid faces found in any input images.")

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
        st.warning("No identity triggers (img1, img2) found in prompt. Using all detected identities.")
        active_slots = list(range(min(len(id_embeds), 2)))

    pipe.unet.set_active_slots(active_slots)

    # Add batch dimension
    id_embeds = id_embeds.unsqueeze(0)
    id_pixel_values = id_pixel_values.unsqueeze(0)

    # Set seed
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Move to device
    id_embeds = id_embeds.to(device=pipe.unet.device, dtype=pipe.unet.dtype)
    identity_bboxes = identity_bboxes.to(device=pipe.unet.device, dtype=pipe.unet.dtype)

    pipe.unet.set_identity_data(id_embeds, identity_bboxes)

    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    start_merge_step = min(start_merge_step, 30)

    # Generate
    images = pipe(
        prompt=styled_prompt,
        width=output_width,
        height=output_height,
        negative_prompt=styled_negative,
        num_images_per_prompt=num_outputs,
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

    return images, seed


# ============================================================================
# Watermark Addition
# ============================================================================

def add_watermark_to_image(model, device, pil_image, corner):
    """Add watermark to a PIL image."""

    if model is None:
        st.error("Watermark model not loaded")
        return pil_image

    # Convert PIL to numpy (RGB)
    img = np.array(pil_image)
    original_h, original_w = img.shape[:2]

    # Get model's expected size
    template = model.template_bottom_right
    image_size = template.shape[1]

    # Resize and normalize
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # HWC -> CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        output, _ = model(img_tensor, [corner])

    # Post-process
    watermarked = output.squeeze(0).cpu().numpy()
    watermarked = np.transpose(watermarked, (1, 2, 0))
    watermarked = np.clip(watermarked * 255, 0, 255).astype(np.uint8)

    # Resize back
    watermarked_original = cv2.resize(watermarked, (original_w, original_h))

    return Image.fromarray(watermarked_original)


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="PhotoMaker + Watermark Pipeline",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 PhotoMaker + Watermark Pipeline")
    st.markdown("""
    Generate personalized images using PhotoMaker V2 and automatically add invisible watermarks.

    **Steps:**
    1. Upload face image(s)
    2. Enter a prompt with `img1` or `img2` trigger words
    3. Generate and watermark!
    """)

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Model Paths")
        watermark_path = st.text_input(
            "Watermark Model Path",
            value=WATERMARK_MODEL_PATH,
            help="Path to the watermark injection model (.pth file)"
        )

        st.subheader("Generation Settings")
        style_name = st.selectbox("Style", STYLE_NAMES, index=STYLE_NAMES.index(DEFAULT_STYLE_NAME))
        aspect_ratio = st.selectbox("Aspect Ratio", ASPECT_RATIO_LABELS, index=0)

        num_steps = st.slider("Inference Steps", 20, 100, 50)
        guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 6.0, 0.5)
        style_strength = st.slider("Style Strength %", 15, 50, 20)

        seed = st.number_input("Seed (0 for random)", min_value=0, max_value=MAX_SEED, value=0)
        if seed == 0:
            seed = None

        st.subheader("Watermark Settings")
        watermark_corner = st.selectbox(
            "Watermark Corner",
            ["random"] + CORNERS,
            index=0
        )
        enable_watermark = st.checkbox("Enable Watermark", value=True)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Input")

        uploaded_files = st.file_uploader(
            "Upload face image(s)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload one or more images containing faces"
        )

        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} image(s)")
            cols = st.columns(min(len(uploaded_files), 3))
            for i, f in enumerate(uploaded_files):
                with cols[i % 3]:
                    st.image(f, caption=f"Image {i+1}", use_container_width=True)

        prompt = st.text_area(
            "Prompt",
            placeholder="a man img1 on the left and a woman img2 on the right",
            help="Use 'img1' and 'img2' as trigger words for identities"
        )

        negative_prompt = st.text_area(
            "Negative Prompt",
            value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry",
            help="What to avoid in the generation"
        )

        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    with col2:
        st.subheader("🖼️ Output")
        output_placeholder = st.empty()

    # Generation logic
    if generate_btn:
        if not uploaded_files:
            st.error("Please upload at least one face image")
            return

        if not prompt:
            st.error("Please enter a prompt")
            return

        # Check for trigger words
        if "img1" not in prompt and "img2" not in prompt:
            st.warning("Prompt should contain 'img1' or 'img2' trigger words for best results")

        with st.spinner("Loading models..."):
            pipe, device = load_photomaker_pipeline()
            face_detector = load_face_detector()

            if enable_watermark:
                # Update watermark path if changed
                global WATERMARK_MODEL_PATH
                WATERMARK_MODEL_PATH = watermark_path
                watermark_model, _ = load_watermark_model()
            else:
                watermark_model = None

        # Convert uploaded files to PIL images
        input_images = []
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            input_images.append(img)

        # Get output dimensions
        output_w, output_h = aspect_ratios[aspect_ratio]

        with st.spinner("Generating with PhotoMaker..."):
            try:
                generated_images, used_seed = generate_photomaker_image(
                    pipe=pipe,
                    face_detector=face_detector,
                    device=device,
                    input_images=input_images,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    style_name=style_name,
                    output_width=output_w,
                    output_height=output_h,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    style_strength_ratio=style_strength,
                    seed=seed,
                    num_outputs=1
                )
                st.success(f"Generated with seed: {used_seed}")
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                return

        # Apply watermark
        if enable_watermark and watermark_model is not None:
            with st.spinner("Adding watermark..."):
                corner = watermark_corner if watermark_corner != "random" else random.choice(CORNERS)
                watermarked_images = []
                for img in generated_images:
                    watermarked = add_watermark_to_image(watermark_model, device, img, corner)
                    watermarked_images.append(watermarked)
                st.info(f"Watermark applied at: {corner}")
        else:
            watermarked_images = generated_images

        # Display results
        with output_placeholder.container():
            st.write("**PhotoMaker Output:**")
            for i, img in enumerate(generated_images):
                st.image(img, caption=f"Generated {i+1}", use_container_width=True)

            if enable_watermark and watermark_model is not None:
                st.write("**Watermarked Output:**")
                for i, img in enumerate(watermarked_images):
                    st.image(img, caption=f"Watermarked {i+1}", use_container_width=True)

            # Download buttons
            st.write("**Download:**")
            for i, img in enumerate(watermarked_images):
                # Save to bytes
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label=f"Download Image {i+1}",
                    data=buf.getvalue(),
                    file_name=f"generated_{used_seed}_{i+1}.png",
                    mime="image/png"
                )


if __name__ == "__main__":
    main()

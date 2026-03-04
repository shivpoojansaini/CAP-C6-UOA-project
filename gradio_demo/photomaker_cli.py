#!/usr/bin/env python3
"""
PhotoMaker V2 CLI - Generate images without Gradio
Just edit the configuration below and run: python photomaker_cli.py
"""
print("🚀🚀🚀 RUNNING PHOTOMAKER_CLI.PY 🚀🚀🚀")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import sys
from pathlib import Path

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces

from style_template import styles
from aspect_ratio_template import aspect_ratios
from insightface.utils import face_align
from PIL import Image
import PIL.Image
from photomaker.identity_prompt_parser import extract_identity_prompt_map_clean
from photomaker.identity_evaluator import evaluate_identities_dynamic


# ============================================================
# CONFIGURATION - Edit these values directly
# ============================================================

# Input image(s) - provide path(s) to face image(s)
INPUT_IMAGES = [
    "/teamspace/studios/this_studio/photomaker-spatial-multi-id-layered/Data/Input/Young&Old_1.png"
]

# Prompt - must include 'img' trigger word
PROMPT = "a man img1 with a man img2"
# Output settings 
OUTPUT_DIR = "/teamspace/studios/this_studio/photomaker-spatial-multi-id-layered/Data/Output"
NUM_OUTPUTS = 2

# Style (check style_template.py for options)
STYLE_NAME = "Photographic (Default)"

# Negative prompt
NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# Output dimensions
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# Generation parameters
NUM_STEPS = 50
GUIDANCE_SCALE = 7.5
STYLE_STRENGTH_RATIO = 20
SEED = None  # Set to None for random seed, or specify a number

# Sketch/Doodle settings (optional)
USE_SKETCH = False
SKETCH_IMAGE_PATH = None  # e.g., "./sketch.png"
ADAPTER_CONDITIONING_SCALE = 0.7
ADAPTER_CONDITIONING_FACTOR = 0.8

# ============================================================
# END OF CONFIGURATION
# ============================================================

MAX_SEED = np.iinfo(np.int32).max

# --------------------------------------------------
# 🔥 Slot Activation Utility
# --------------------------------------------------

def compute_active_slots(prompt, trigger_words):
    active_slots = []

    for i, word in enumerate(trigger_words):
        if word in prompt:
            active_slots.append(i)

    return active_slots

def get_device():
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
    default_style = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default_style])
    return p.replace("{prompt}", positive), n + ' ' + negative

from diffusers.models.attention_processor import LoRAAttnProcessor2_0

from diffusers.models.attention_processor import AttnProcessor2_0




def load_pipeline(device):
    print("Loading pipeline...\n")

    base_model_path = "SG161222/RealVisXL_V4.0"

    # --------------------------------------------------
    # 1️⃣ Torch dtype selection
    # --------------------------------------------------
    torch_dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    if device == "mps":
        torch_dtype = torch.float16

    # --------------------------------------------------
    # 2️⃣ Load T2I Adapter
    # --------------------------------------------------
    print("Loading T2I adapter...")
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        torch_dtype=torch_dtype,
        variant="fp16",
    ).to(device)

    # --------------------------------------------------
    # 3️⃣ Load Base Pipeline
    # --------------------------------------------------
    print("Loading main pipeline...")
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_path,
        adapter=adapter,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    # --------------------------------------------------
    # 4️⃣ Load PhotoMaker Adapter (LoRA + ID encoder)
    # --------------------------------------------------
    print("Loading PhotoMaker adapter...")
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
    print("\n🔥 APPLYING CONTROLLED LORA SCALE x1.5")

    for name, module in pipe.unet.named_modules():
        if hasattr(module, "scale"):
            module.scale = 1.0

    print("\n===== LORA NORM CHECK =====")
    for name, param in pipe.unet.named_parameters():
        if "photomaker" in name and "weight" in name:
            print(name, torch.norm(param).item())
            break



    pipe.id_encoder.to(device)

    # ==================================================
    # 🔍 5️⃣ LoRA Module Inspection
    # ==================================================
    print("\n===== LORA MODULE INSPECTION =====")

    found_lora = False
    for name, module in pipe.unet.named_modules():
        if "lora" in name.lower():
            print("LoRA module:", name, "->", type(module))
            found_lora = True

    if not found_lora:
        print("❌ No LoRA modules found in UNet.")

    # ==================================================
    # 🔍 6️⃣ LoRA Parameter Inspection
    # ==================================================
    print("\n===== LORA PARAMETER INSPECTION =====")

    found_lora_params = False
    for name, param in pipe.unet.named_parameters():
        if "lora" in name.lower():
            print("LoRA param:", name, param.shape)
            found_lora_params = True

    if not found_lora_params:
        print("❌ No LoRA parameters found in UNet.")

    # ==================================================
    # 🔍 7️⃣ Attention Processor Audit
    # ==================================================
    print("\n===== ATTENTION PROCESSOR AUDIT =====")
    for name, proc in pipe.unet.attn_processors.items():
        print(name, "->", type(proc).__name__)

    # --------------------------------------------------
    # 8️⃣ Scheduler
    # --------------------------------------------------
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # ==================================================
    # 🔥 9️⃣ Wrap UNet with Identity Slot Injection
    # ==================================================
    print("\n🔥 Wrapping UNet with Identity Slot Injection...")

    from photomaker.identity_slot_unet import IdentitySlotUNet

    pipe.unet = IdentitySlotUNet(pipe.unet)
    pipe.unet.disable_injection = True


    print("✅ UNet successfully wrapped.")

    # Final device push
    pipe.to(device)

    print("\n🔍 feature_extractor:", pipe.feature_extractor)
    print("🔍 image_processor:", pipe.image_processor)

    print("\nPipeline loaded successfully!\n")

    return pipe



def load_face_detector(device):
    print("Loading face detector...")

    # -------------------------------------------------
    # Select ONNX providers
    # -------------------------------------------------
    if device == "cuda":
        providers = ["CUDAExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1  # required for CPU in InsightFace

    # -------------------------------------------------
    # Initialize FaceAnalysis
    # -------------------------------------------------
    face_detector = FaceAnalysis2(
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )

    # -------------------------------------------------
    # Strong multi-face configuration
    # -------------------------------------------------
    face_detector.prepare(
        ctx_id=ctx_id,
        det_size=(1024, 1024),   # higher resolution improves multi-face recall
        det_thresh=0.25          # lower threshold → detect tightly grouped faces
    )

    print("Face detector loaded successfully.")
    return face_detector




def generate_image(pipe, face_detector, device):
    # Handle sketch input
    sketch_image = None
    adapter_scale = 0.
    adapter_factor = 0.
    
    if USE_SKETCH and SKETCH_IMAGE_PATH:
        from PIL import Image
        sketch_image = Image.open(SKETCH_IMAGE_PATH).convert("RGBA")
        r, g, b, a = sketch_image.split()
        sketch_image = a.convert("RGB")
        sketch_image = TF.to_tensor(sketch_image) > 0.5
        sketch_image = TF.to_pil_image(sketch_image.to(torch.float32))
        adapter_scale = ADAPTER_CONDITIONING_SCALE
        adapter_factor = ADAPTER_CONDITIONING_FACTOR

   



    output_w = OUTPUT_WIDTH or 1024
    output_h = OUTPUT_HEIGHT or 1024
    
    print(f"[Info] Output dimensions: {output_w} x {output_h}")

    # Apply style
    prompt, negative_prompt = apply_style(STYLE_NAME, PROMPT, NEGATIVE_PROMPT)

    # --------------------------------------------------
    # Load input images
    # --------------------------------------------------
    if not INPUT_IMAGES:
        raise ValueError("No input images! Edit INPUT_IMAGES in the script.")

    input_id_images = []
    for img_path in INPUT_IMAGES:
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        input_id_images.append(load_image(img_path))

    
    # --------------------------------------------------
    # Load input images (original images)
    # --------------------------------------------------
    original_images = []

    for img_path in INPUT_IMAGES:
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        original_images.append(load_image(img_path))

    print("🔥 Loaded original images:", len(original_images))


    # --------------------------------------------------
    # Extract face embeddings (TRUE MULTI-FACE SUPPORT)
    # --------------------------------------------------

    id_embed_list = []
    id_pixel_list = []
    input_id_images = []
    bbox_list = []

    for img_idx, img in enumerate(original_images):

        img_array = np.array(img)[:, :, ::-1]  # RGB → BGR

        # 🔥 Use robust multi-face analyzer
        faces = analyze_faces(face_detector, img_array)

        if len(faces) == 0:
            print(f"⚠️ No face detected in image {img_idx}")
            continue

        # 🔥 Force deterministic left → right ordering
        faces = sorted(faces, key=lambda f: f.bbox[0])

        print(f"\n🔥 Detected {len(faces)} face(s) in image {img_idx}")
        print("Original image size:", img.width, img.height)

        for face_idx, face in enumerate(faces):

            print(f"\n🔥 Processing face {face_idx} in image {img_idx}")
            print("Raw bbox:", face.bbox)

            # --------------------------------------------------
            # 1️⃣ Normalize bounding box
            # --------------------------------------------------
            x1, y1, x2, y2 = face.bbox

            W = img.width
            H = img.height

            x1 /= W
            x2 /= W
            y1 /= H
            y2 /= H

            normalized_bbox = [x1, y1, x2, y2]
            bbox_list.append(normalized_bbox)

            print("Normalized bbox:", normalized_bbox)

            # --------------------------------------------------
            # 2️⃣ Extract embedding
            # --------------------------------------------------
            raw_embedding = face["embedding"]

            print("   🔍 Embedding shape:", raw_embedding.shape)
            print("   🔍 Norm:", np.linalg.norm(raw_embedding))

            id_embed_list.append(torch.from_numpy(raw_embedding).float())

            # --------------------------------------------------
            # 3️⃣ Align face to 224x224
            # --------------------------------------------------
            aligned_face = face_align.norm_crop(
                img_array,
                landmark=face.kps,
                image_size=224
            )

            aligned_face = aligned_face[:, :, ::-1]  # BGR → RGB
            aligned_face = PIL.Image.fromarray(aligned_face)

            # --------------------------------------------------
            # 4️⃣ Preprocess for SDXL
            # --------------------------------------------------
            pixel_tensor = pipe.image_processor.preprocess(aligned_face)[0]

            print("   🔍 Pixel tensor shape:", pixel_tensor.shape)

            id_pixel_list.append(pixel_tensor)
            input_id_images.append(aligned_face)


    

    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------
    if len(id_embed_list) == 0:
        raise ValueError("❌ No valid faces found in any input images.")

    import torch.nn.functional as F

    # --------------------------------------------------
    # 🔥 ARC-FACE BASED IDENTITY GROUPING ACROSS IMAGES
    # --------------------------------------------------

    # Stack all detected face embeddings
    all_embeds = torch.stack(id_embed_list)  # [N_total_faces, 512]
    all_embeds = F.normalize(all_embeds, dim=1)

    print("\n🔥 Total detected faces:", all_embeds.shape[0])

    groups = []
    threshold = 0.5  # tune between 0.45–0.6

    for idx, embed in enumerate(all_embeds):

        placed = False

        for group in groups:
            ref_idx = group[0]
            similarity = torch.dot(embed, all_embeds[ref_idx]).item()

            if similarity > threshold:
                group.append(idx)
                placed = True
                break

        if not placed:
            groups.append([idx])

    print("🔥 Number of identity groups found:", len(groups))

    # --------------------------------------------------
    # 🔥 AVERAGE EMBEDDINGS PER IDENTITY
    # --------------------------------------------------

    final_embeds = []
    final_pixels = []
    final_bboxes = []

    for group_idx, group in enumerate(groups):

        group_embeds = torch.stack([all_embeds[i] for i in group])
        avg_embed = group_embeds.mean(dim=0)
        avg_embed = torch.nn.functional.normalize(avg_embed, dim=0)


        final_embeds.append(avg_embed)

        # Use first occurrence for pixel + bbox
        final_pixels.append(id_pixel_list[group[0]])
        final_bboxes.append(bbox_list[group[0]])

        print(f"   Identity {group_idx} → {len(group)} face(s) grouped")

    # Stack final identities
    id_embeds = torch.stack(final_embeds)              # [N_id, 512]
    id_pixel_values = torch.stack(final_pixels)       # [N_id, 3, 224, 224]
    identity_bboxes = torch.tensor(final_bboxes).float()

    # --------------------------------------------------
    # 🔥 FORCE LEFT → RIGHT IDENTITY ORDERING
    # --------------------------------------------------

    # Compute x-center for each identity bbox
    x_centers = []

    for bbox in identity_bboxes:
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2.0
        x_centers.append(x_center.item())

    # Sort identities by x-center (left → right)
    sorted_indices = sorted(range(len(x_centers)), key=lambda i: x_centers[i])

    print("🔥 Reordering identities left → right:", sorted_indices)

    # Reorder everything consistently
    identity_bboxes = identity_bboxes[sorted_indices]
    id_embeds = id_embeds[sorted_indices]
    id_pixel_values = id_pixel_values[sorted_indices]
    
    for i, bbox in enumerate(identity_bboxes):
        x1, y1, x2, y2 = bbox
        print(f"Identity {i} x-center:", (x1 + x2)/2)

    # Save reference embeddings for evaluation
    reference_arcface_embeds = id_embeds.clone().detach()

    print("\n🔥 Final id_embeds shape:", id_embeds.shape)
    print("🔥 Final id_pixel_values shape:", id_pixel_values.shape)
    print("🔥 Final identity_bboxes shape:", identity_bboxes.shape)

    # --------------------------------------------------
    # 🔥 Multi-trigger validation (AFTER grouping)
    # --------------------------------------------------

    num_identities = id_embeds.shape[0]
    active_triggers = pipe.trigger_words[:num_identities]

    trigger_token_ids = [
        pipe.tokenizer.convert_tokens_to_ids(t)
        for t in active_triggers
    ]

    input_ids = pipe.tokenizer.encode(prompt)

    print("🔍 Trigger words:", active_triggers)

    # --------------------------------------------------
    # 🔥 Flexible Trigger Activation (NON-STRICT)
    # --------------------------------------------------

    num_identities = id_embeds.shape[0]
    active_triggers = pipe.trigger_words[:num_identities]

    trigger_token_ids = [
        pipe.tokenizer.convert_tokens_to_ids(t)
        for t in active_triggers
    ]

    input_ids = pipe.tokenizer.encode(prompt)

    print("🔍 Trigger words:", active_triggers)

    active_slots = []

    for i, (trigger_word, trigger_id) in enumerate(zip(active_triggers, trigger_token_ids)):

        count = input_ids.count(trigger_id)

        print(f"🔍 '{trigger_word}' appears {count} time(s)")

        if count > 1:
            raise ValueError(
                f"Trigger word '{trigger_word}' appears multiple times."
            )

        if count == 1:
            active_slots.append(i)

    print("✅ Active identity slots:", active_slots)

    # 🔥 Provide active slot mask to UNet
    pipe.unet.active_slots = active_slots


    # --------------------------------------------------
    # Add batch dimension (VERY IMPORTANT)
    # --------------------------------------------------

    id_embeds = id_embeds.unsqueeze(0)              # [1, N_id, 512]
    id_pixel_values = id_pixel_values.unsqueeze(0)  # [1, N_id, 3, 224, 224]

    print("🔥 After unsqueeze id_embeds:", id_embeds.shape)
    print("🔥 After unsqueeze id_pixel_values:", id_pixel_values.shape)

    # --------------------------------------------------
    # Validate alignment
    # --------------------------------------------------

    assert id_embeds.shape[1] == id_pixel_values.shape[1], \
        "Mismatch between identity embeddings and pixel tensors!"

    print("✅ Identity count:", id_embeds.shape[1])
    print("✅ Embedding dimension:", id_embeds.shape[2])



    # --------------------------------------------------
    # Handle seed
    # --------------------------------------------------
    seed = SEED if SEED is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)


    print("Starting inference...")
    print(f"[Info] Seed: {seed}")
    print(f"[Info] Prompt: {prompt}")
    
    start_merge_step = int(float(STYLE_STRENGTH_RATIO) / 100 * NUM_STEPS)
    if start_merge_step > 30:
        start_merge_step = 30
    

    # --------------------------------------------------
    # 🔥 BUILD SPATIAL MASKS
    # --------------------------------------------------

    num_identities = identity_bboxes.shape[0]
    H_mask = 128
    W_mask = 128

    base_masks = torch.zeros((num_identities, H_mask, W_mask))

    for i, bbox in enumerate(identity_bboxes):
        x1, y1, x2, y2 = bbox

        x1_i = int(x1 * W_mask)
        x2_i = int(x2 * W_mask)
        y1_i = int(y1 * H_mask)
        y2_i = int(y2 * H_mask)

        base_masks[i, y1_i:y2_i, x1_i:x2_i] = 1.0

    print("🔥 Built base_masks:", base_masks.shape)
    
    # -----------------------------------------
    # 🔥 SORT IDENTITIES LEFT → RIGHT
    # -----------------------------------------

    # identity_bboxes shape: [N, 4]
    # id_embeds shape: [B, N, 512]  OR  [N, 512]

    # If embeddings are [N, 512]
    if id_embeds.dim() == 2:
        centers = (identity_bboxes[:, 0] + identity_bboxes[:, 2]) / 2
        sorted_indices = torch.argsort(centers)

        identity_bboxes = identity_bboxes[sorted_indices]
        id_embeds = id_embeds[sorted_indices]

    # If embeddings are [B, N, 512]
    elif id_embeds.dim() == 3:
        centers = (identity_bboxes[:, 0] + identity_bboxes[:, 2]) / 2
        sorted_indices = torch.argsort(centers)

        identity_bboxes = identity_bboxes[sorted_indices]
        id_embeds = id_embeds[:, sorted_indices, :]

    # -----------------------------------------
    # 🔥 Compute active slots from prompt
    # -----------------------------------------

    trigger_words = ["img1", "img2"]   # must match your system

    active_slots = compute_active_slots(PROMPT, trigger_words)
    if len(active_slots) == 0:
        print("❌ No identity trigger found in prompt.")
    else:
        print("🔥 Active slots:", active_slots)

    if hasattr(pipe.unet, "set_active_slots"):
        pipe.unet.set_active_slots(active_slots)

    

    # -----------------------------------------
    # 🔥 Ensure correct device + dtype
    # -----------------------------------------

    id_embeds = id_embeds.to(pipe.unet.device)
    id_embeds = id_embeds.to(pipe.unet.dtype)

    identity_bboxes = identity_bboxes.to(pipe.unet.device)
    identity_bboxes = identity_bboxes.to(pipe.unet.dtype)

    print("🔎 Sorted BBoxes:", identity_bboxes)
    print("🔎 id_embeds shape:", id_embeds.shape)
    print("🔎 id_embeds device:", id_embeds.device)
    print("🔎 id_embeds dtype:", id_embeds.dtype)

    # -----------------------------------------
    # 🔥 Provide identity data to UNet wrapper
    # -----------------------------------------

    if hasattr(pipe.unet, "set_identity_data"):
        pipe.unet.set_identity_data(id_embeds, identity_bboxes)


    images = pipe(
        prompt=prompt,
        width=output_w,
        height=output_h,
        negative_prompt=negative_prompt,
        num_images_per_prompt=NUM_OUTPUTS,
        num_inference_steps=NUM_STEPS,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=GUIDANCE_SCALE,

        # ✅ Pass raw PIL images
        input_id_images=input_id_images,

        # ✅ Pass stacked identity embeddings
        id_embeds=id_embeds,
        id_pixel_values=id_pixel_values,
        image=sketch_image,
        adapter_conditioning_scale=adapter_scale,
        adapter_conditioning_factor=adapter_factor,
        identity_bboxes=identity_bboxes,  # 🔥 pass here
    ).images

    print("\n🔍 ArcFace Identity Fidelity Evaluation")

    # Store similarity results for each generated image
    similarity_results = []

    for img_idx, gen_img in enumerate(images):

        print(f"\n🖼 Generated Image {img_idx}")

        gen_array = np.array(gen_img)[:, :, ::-1]  # RGB → BGR
        gen_faces = analyze_faces(face_detector, gen_array)

        img_result = {"image_idx": img_idx, "assignments": [], "error": None}

        if len(gen_faces) == 0:
            print("❌ No face detected in generated image")
            img_result["error"] = "No face detected in generated image"
            similarity_results.append(img_result)
            continue

        # ✅ Force left → right order
        gen_faces = sorted(gen_faces, key=lambda f: f.bbox[0])
        # 🔎 DEBUG: Print bounding boxes
        for i, face in enumerate(gen_faces):
            print(f"Face {i} bbox:", face.bbox)
        # -------------------------------------------------
        # Collect embeddings for all detected faces
        # -------------------------------------------------
        face_embeds = []

        for gen_face in gen_faces:
            emb = torch.from_numpy(gen_face["embedding"]).float()
            face_embeds.append(emb)

        face_embeds = torch.stack(face_embeds)  # [num_faces, 512]

        # -------------------------------------------------
        # Select only active reference embeddings
        # -------------------------------------------------

        ref_embeds = torch.stack(
            [reference_arcface_embeds[i] for i in active_slots]
        ).to(face_embeds.device)  # shape: [num_active_ids, 512]


        # -------------------------------------------------
        # Dynamic Assignment
        # -------------------------------------------------
        assignments, sim_matrix = evaluate_identities_dynamic(
            face_embeds,
            ref_embeds
        )

        # -------------------------------------------------
        # Calculate Direct and Cross Matching Scores
        # -------------------------------------------------
        num_faces = min(face_embeds.shape[0], ref_embeds.shape[0])
        sim_matrix_np = sim_matrix.cpu().numpy()

        # Direct matching: face[i] <-> ref[i]
        direct_scores = []
        for i in range(num_faces):
            if i < sim_matrix_np.shape[0] and i < sim_matrix_np.shape[1]:
                direct_scores.append(float(sim_matrix_np[i, i]))

        # Cross matching: face[i] <-> ref[num_faces-1-i]
        cross_scores = []
        for i in range(num_faces):
            j = num_faces - 1 - i
            if i < sim_matrix_np.shape[0] and j < sim_matrix_np.shape[1]:
                cross_scores.append(float(sim_matrix_np[i, j]))

        direct_avg = sum(direct_scores) / len(direct_scores) if direct_scores else 0
        cross_avg = sum(cross_scores) / len(cross_scores) if cross_scores else 0

        print(f"\n📊 Direct Matching (positional): {direct_scores} -> Avg: {direct_avg:.4f}")
        print(f"📊 Cross Matching (swapped): {cross_scores} -> Avg: {cross_avg:.4f}")

        # Store both matching modes
        img_result["direct_matching"] = {
            "scores": direct_scores,
            "average": direct_avg
        }
        img_result["cross_matching"] = {
            "scores": cross_scores,
            "average": cross_avg
        }
        img_result["best_mode"] = "direct" if direct_avg >= cross_avg else "cross"
        img_result["best_average"] = max(direct_avg, cross_avg)

        # -------------------------------------------------
        # Pretty Output
        # -------------------------------------------------
        print("\n📊 Final Identity Assignment (Optimal):")
        for face_idx, assigned_id, score in assignments:
            print(
                f"   ✅ Face {face_idx} → Identity {assigned_id} "
                f"(Cosine: {score:.4f})"
            )
            img_result["assignments"].append({
                "face_idx": face_idx,
                "identity_idx": assigned_id,
                "similarity": float(score)
            })

        similarity_results.append(img_result)


    # -----------------------------------------
    # 🔥 CLEAR SLOT INJECTION STATE
    # -----------------------------------------

    
    if hasattr(pipe.unet, "clear_identity_data"):
        pipe.unet.clear_identity_data()
        print("🧹 Cleared identity slot data from UNet")

    return images, seed, similarity_results


def main():
    print("=" * 50)
    print("PhotoMaker V2 CLI")
    print("=" * 50)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)
    # 🔥 Experiment toggles
    pipe.enable_routing = True          # disable attention routing
    pipe.enable_slot_injection = False    # keep additive slot injection
    pipe.face_detector = face_detector
    try:
        images, used_seed, similarity_results = generate_image(pipe, face_detector, device)

        print(f"\nSaving {len(images)} image(s) to {output_dir}/")
        for i, img in enumerate(images):
            filename = f"output_seed{used_seed}_{i+1}.png"
            filepath = output_dir / filename
            img.save(filepath)
            print(f"  Saved: {filepath}")

        print(f"\nDone! Generated {len(images)} image(s) with seed {used_seed}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

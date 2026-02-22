#!/usr/bin/env python3
"""
PhotoMaker V2 — 2 Identity CLI (FINAL, Adapter Pipeline)
"""

import os
import random
import torch
import numpy as np
from pathlib import Path

from diffusers import (
    EulerDiscreteScheduler,
    T2IAdapter,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

from photomaker import (
    PhotoMakerStableDiffusionXLAdapterPipeline,
    FaceAnalysis2,
    analyze_faces,
)

# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGES = [
    "/teamspace/studios/this_studio/PhotoMaker/Data/Input/person1.jpg",
    "/teamspace/studios/this_studio/PhotoMaker/Data/Input/person2.jpg",
]

OUTPUT_DIR = "/teamspace/studios/this_studio/PhotoMaker/Data/Output"

PROMPT = (
    "cinematic photo of two men img standing together, "
    "holding a tennis trophy, professional lighting, 35mm photograph"
)

NEGATIVE_PROMPT = "lowres, bad anatomy, blurry, watermark"

BASE_MODEL = "SG161222/RealVisXL_V4.0"
PHOTOMAKER_REPO = "TencentARC/PhotoMaker-V2"
PHOTOMAKER_WEIGHTS = "photomaker-v2.bin"

T2I_ADAPTER_REPO = "TencentARC/t2i-adapter-sketch-sdxl-1.0"

WIDTH = 1024
HEIGHT = 1024
NUM_IMAGES = 2
NUM_STEPS = 40
GUIDANCE_SCALE = 5.0
STYLE_STRENGTH_RATIO = 20
SEED = None

# ============================================================
# HELPERS
# ============================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_face_detector(device):
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    detector = FaceAnalysis2(
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector


def extract_identity_embeddings(detector, images):
    embeds = []
    for img in images:
        arr = np.array(img)[:, :, ::-1]  # RGB → BGR
        faces = analyze_faces(detector, arr)
        if len(faces) == 0:
            raise ValueError("No face detected in one image.")
        embeds.append(torch.from_numpy(faces[0]["embedding"]))
    return torch.stack(embeds)  # [N, 512]


def load_pipeline(device):
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("[INFO] Loading T2I Adapter...")
    adapter = T2IAdapter.from_pretrained(
        T2I_ADAPTER_REPO,
        torch_dtype=torch_dtype,
        variant="fp16",
    ).to(device)

    print("[INFO] Loading PhotoMaker Adapter Pipeline...")
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        BASE_MODEL,
        adapter=adapter,
        torch_dtype=torch_dtype,
        variant="fp16",
        use_safetensors=True,
    )

    ckpt_path = hf_hub_download(
        repo_id=PHOTOMAKER_REPO,
        filename=PHOTOMAKER_WEIGHTS,
        repo_type="model",
    )

    pipe.load_photomaker_adapter(
        os.path.dirname(ckpt_path),
        weight_name=os.path.basename(ckpt_path),
        trigger_word="img",
        pm_version="v2",
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()
    pipe.to(device)

    return pipe


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PhotoMaker V2 — 2 Identity FINAL CLI")
    print("=" * 60)

    device = get_device()
    print(f"[INFO] Device: {device}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_id_images = [load_image(p) for p in INPUT_IMAGES]

    face_detector = load_face_detector(device)
    id_embeds = extract_identity_embeddings(face_detector, input_id_images).to(device)

    print(f"[INFO] Number of identities: {id_embeds.shape[0]}")

    pipe = load_pipeline(device)

    seed = SEED if SEED is not None else random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    start_merge_step = min(int((STYLE_STRENGTH_RATIO / 100) * NUM_STEPS), 30)

    print("[INFO] Generating...")
    images = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_images_per_prompt=NUM_IMAGES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        start_merge_step=start_merge_step,

        # 🔑 REQUIRED
        input_id_images=input_id_images,
        id_embeds=id_embeds,
    ).images

    for i, img in enumerate(images):
        out = output_dir / f"output_seed{seed}_{i+1}.png"
        img.save(out)
        print(f"[SAVED] {out}")

    print("[DONE]")


if __name__ == "__main__":
    main()

import os
import onnxruntime as ort

# Detect CUDA availability through ONNXRuntime BEFORE importing insightface
cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()

if cuda_available:
    print("GPU detected — enabling CUDA for InsightFace")
    os.environ["INSIGHTFACE_DISABLE_CUDA"] = "0"
else:
    print("No GPU detected — forcing InsightFace to CPU")
    os.environ["INSIGHTFACE_DISABLE_CUDA"] = "1"


import gradio as gr
import cv2
import numpy as np
import insightface
from numpy.linalg import norm

# -----------------------------
# Load ArcFace model once
# -----------------------------
_arcface_app = None

def load_arcface():
    global _arcface_app
    if _arcface_app is not None:
        return _arcface_app

    try:
        _arcface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _arcface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("ArcFace: GPU enabled")
    except Exception:
        print("ArcFace: GPU unavailable, using CPU")
        _arcface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        _arcface_app.prepare(ctx_id=-1, det_size=(640, 640))

    return _arcface_app


# -----------------------------
# Utility: PIL → BGR
# -----------------------------
def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# -----------------------------
# Detect faces and sort left→right
# -----------------------------
def detect_and_sort_faces(img_bgr, expected_faces=2):
    app = load_arcface()
    faces = app.get(img_bgr)

    if len(faces) < expected_faces:
        return None  # return None instead of crashing

    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])
    return faces_sorted[:expected_faces]


# -----------------------------
# Extract embedding
# -----------------------------
def extract_embedding(face):
    return face.normed_embedding.astype(np.float32)

# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8))

# -----------------------------
# Main comparison function
# -----------------------------
def compare_faces(img1, img2, comparison_mode):
    img1_bgr = pil_to_bgr(img1)
    img2_bgr = pil_to_bgr(img2)

    faces1 = detect_and_sort_faces(img1_bgr, expected_faces=2)
    faces2 = detect_and_sort_faces(img2_bgr, expected_faces=2)

    if faces1 is None or faces2 is None:
        return "❌ Each image must contain TWO faces for dual-identity mode."

    left1, right1 = faces1
    left2, right2 = faces2

    emb_left1 = extract_embedding(left1)
    emb_right1 = extract_embedding(right1)
    emb_left2 = extract_embedding(left2)
    emb_right2 = extract_embedding(right2)

    if comparison_mode == "Left₁–Left₂ & Right₁–Right₂":
        return (
            f"Left₁ ↔ Left₂: {cosine_similarity(emb_left1, emb_left2):.4f}\n"
            f"Right₁ ↔ Right₂: {cosine_similarity(emb_right1, emb_right2):.4f}"
        )
    else:
        return (
            f"Left₁ ↔ Right₂: {cosine_similarity(emb_left1, emb_right2):.4f}\n"
            f"Right₁ ↔ Left₂: {cosine_similarity(emb_right1, emb_left2):.4f}"
        )


# -----------------------------
# Single Identity Comparison
# -----------------------------
def compare_single_identity(img1, img2):
    img1_bgr = pil_to_bgr(img1)
    img2_bgr = pil_to_bgr(img2)

    faces1 = detect_and_sort_faces(img1_bgr, expected_faces=1)
    faces2 = detect_and_sort_faces(img2_bgr, expected_faces=1)

    if faces1 is None or faces2 is None:
        return "❌ Each image must contain at least ONE face."

    emb1 = extract_embedding(faces1[0])
    emb2 = extract_embedding(faces2[0])

    score = cosine_similarity(emb1, emb2)
    return f"Single Identity Similarity: {score:.4f}"



# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 ArcFace Identity Similarity — Single & Dual Identity Modes")

    # Identity mode selector
    identity_mode = gr.Radio(
        ["Single Identity", "Dual Identity"],
        value="Single Identity",
        label="Identity Mode"
    )

    # Row for the two input images
    with gr.Row():
        img1 = gr.Image(label="Input Image 1")
        img2 = gr.Image(label="Input Image 2")

    # Comparison mode (only for dual identity)
    comparison_mode = gr.Radio(
        ["Left₁–Left₂ & Right₁–Right₂", "Left₁–Right₂ & Right₁–Left₂"],
        value="Left₁–Left₂ & Right₁–Right₂",
        label="Comparison Mode",
        interactive=True,
        visible=False  # hidden by default
    )

    # Result box next to comparison mode
    with gr.Row():
        output = gr.Textbox(label="Similarity Results", interactive=False)

    btn = gr.Button("Compute Similarity")

    # --- Logic to show/hide comparison options ---
    def update_visibility(mode):
        if mode == "Single Identity":
            return gr.update(visible=False, interactive=False)
        else:
            return gr.update(visible=True, interactive=True)

    identity_mode.change(
        update_visibility,
        inputs=identity_mode,
        outputs=comparison_mode
    )

    # --- Main compute function ---
    def compute(img1_in, img2_in, mode, comp_mode):
        if mode == "Single Identity":
            # Single-face comparison
            return compare_single_identity(img1_in, img2_in)
        else:
            # Dual-face comparison
            return compare_faces(img1_in, img2_in, comp_mode)

    btn.click(
        compute,
        inputs=[img1, img2, identity_mode, comparison_mode],
        outputs=output
    )

demo.launch(share=True)




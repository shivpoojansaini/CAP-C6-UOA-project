import gradio as gr
from PIL import Image
import torch
import numpy as np
import cv2
import math
import os
import tempfile
import zipfile
from skimage.metrics import structural_similarity as ssim

from inference import (
    WatermarkInjectionNetwork,
    AlphaEncoder,
    AlphaDecoder,
    load_model,
    add_watermark,
    CORNERS,
    MODEL_PATH
)

from photomaker_cli import (
    load_pipeline,
    load_face_detector,
    generate_image,
    get_device
)

# -------------------------------------------------
# Load models once
# -------------------------------------------------
device = get_device()
pipe = load_pipeline(device)
face_detector = load_face_detector(device)
pipe.face_detector = face_detector

if not hasattr(pipe, "enable_routing"):
    pipe.enable_routing = True
if not hasattr(pipe, "enable_slot_injection"):
    pipe.enable_slot_injection = False

wm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global watermark model state
wm_model = None
wm_model_path = None


def load_watermark_model(model_path):
    """Load watermark model from specified path."""
    global wm_model, wm_model_path

    if not model_path or not model_path.strip():
        return "Please enter a model path", False

    model_path = model_path.strip()

    if not os.path.exists(model_path):
        return f"Model file not found: {model_path}", False

    try:
        wm_model = load_model(model_path, wm_device)
        wm_model_path = model_path
        return f"Model loaded successfully from: {model_path}", True
    except Exception as e:
        return f"Failed to load model: {str(e)}", False


# Try loading default model on startup
if os.path.exists(MODEL_PATH):
    wm_model = load_model(MODEL_PATH, wm_device)
    wm_model_path = MODEL_PATH
    print(f"Watermark model loaded from: {MODEL_PATH}")
else:
    print(f"Watermark model not found at: {MODEL_PATH}")
    print("Set model path in UI or via WATERMARK_MODEL_PATH environment variable.")


# -------------------------------------------------
# Format ArcFace similarity results
# -------------------------------------------------
def format_similarity_results(similarity_results):
    if not similarity_results:
        return "No similarity data available"

    output_lines = []
    for result in similarity_results:
        img_idx = result["image_idx"]
        output_lines.append(f"Image {img_idx + 1}:")

        if result.get("error"):
            output_lines.append(f"  {result['error']}")
        elif result.get("assignments"):
            for assignment in result["assignments"]:
                face_idx = assignment["face_idx"]
                identity_idx = assignment["identity_idx"]
                similarity = assignment["similarity"]
                # Classify similarity score
                if similarity >= 0.6:
                    quality = "Excellent"
                elif similarity >= 0.45:
                    quality = "Good"
                elif similarity >= 0.3:
                    quality = "Fair"
                else:
                    quality = "Low"
                output_lines.append(
                    f"  Face {face_idx + 1} -> Identity {identity_idx + 1}: "
                    f"{similarity:.4f} ({quality})"
                )
        else:
            output_lines.append("  No faces matched")

        output_lines.append("")

    return "\n".join(output_lines)


# -------------------------------------------------
# PhotoMaker wrapper with progress indicator
# -------------------------------------------------
def run_photomaker(image, prompt, progress=gr.Progress()):
    import photomaker_cli

    if image is None:
        return [], "Please upload an input image", [], None

    if not prompt or not prompt.strip():
        return [], "Please enter a prompt with identity triggers (e.g., 'a photo of img1')", [], None

    progress(0, desc="Preparing input...")
    temp_path = "/tmp/pm_input.png"
    image.save(temp_path)

    photomaker_cli.INPUT_IMAGES = [temp_path]
    photomaker_cli.PROMPT = prompt
    photomaker_cli.NEGATIVE_PROMPT = (
        "nsfw, lowres, bad anatomy, bad hands, text, error"
    )

    try:
        progress(0.1, desc="Detecting faces...")
        progress(0.2, desc="Generating images (this may take a while)...")
        images, seed, similarity_results = generate_image(pipe, face_detector, device)
        progress(0.9, desc="Processing results...")
        pil_images = list(images)
        similarity_text = format_similarity_results(similarity_results)

        # Create downloadable zip file
        progress(0.95, desc="Preparing download...")
        download_path = create_download_zip(pil_images, "generated")

        progress(1.0, desc="Done!")
        return pil_images, similarity_text, pil_images, download_path
    except Exception as e:
        return [], f"Error: {str(e)}", [], None


# -------------------------------------------------
# Create downloadable zip file
# -------------------------------------------------
def create_download_zip(images, prefix="images"):
    if not images:
        return None

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, f"{prefix}_output.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"{prefix}_{i+1}.png")
            img.save(img_path)
            zipf.write(img_path, f"{prefix}_{i+1}.png")

    return zip_path


# -------------------------------------------------
# PSNR + SSIM
# -------------------------------------------------
def compute_psnr_ssim(original_rgb: np.ndarray, watermarked_rgb: np.ndarray):
    orig = original_rgb.astype(np.float32)
    wm = watermarked_rgb.astype(np.float32)

    mse = np.mean((orig - wm) ** 2)
    if mse == 0:
        psnr_value = float("inf")
    else:
        psnr_value = 20 * math.log10(255.0 / math.sqrt(mse))

    ssim_value = ssim(orig, wm, data_range=255, channel_axis=2)

    return psnr_value, ssim_value


# -------------------------------------------------
# Core watermark processing (numpy RGB)
# -------------------------------------------------
def process_watermark_np(image_rgb: np.ndarray, corner: str, strength: float):
    original_h, original_w = image_rgb.shape[:2]

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    template_ref = wm_model.get_template("bottom-right")
    image_size = template_ref.shape[1]

    img_resized = cv2.resize(img_bgr, (image_size, image_size))
    img_norm = img_resized.astype(np.float32) / 255.0

    img_rgb_resized = img_norm[:, :, ::-1].copy()
    img_chw = np.transpose(img_rgb_resized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(wm_device)

    with torch.no_grad():
        _, alpha_base = wm_model(img_tensor, [corner])

    alpha_base = alpha_base[0]
    strength = float(np.clip(strength, 0.0, 1.0))
    alpha_eff = alpha_base * strength

    template = wm_model.get_template(corner).to(wm_device)
    if template.shape[1:] != alpha_eff.shape[1:]:
        template = torch.nn.functional.interpolate(
            template.unsqueeze(0),
            size=alpha_eff.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    image_tensor = img_tensor[0]
    wm_tensor = image_tensor * (1 - alpha_eff) + template * alpha_eff

    wm_np = wm_tensor.cpu().numpy()
    wm_np = np.transpose(wm_np, (1, 2, 0))
    wm_np = np.clip(wm_np * 255, 0, 255).astype(np.uint8)
    wm_np = cv2.cvtColor(wm_np, cv2.COLOR_RGB2BGR)
    wm_np = cv2.cvtColor(wm_np, cv2.COLOR_BGR2RGB)
    wm_np = cv2.resize(wm_np, (original_w, original_h))

    alpha_np = alpha_eff.squeeze(0).cpu().numpy()
    alpha_np = cv2.resize(alpha_np, (original_w, original_h))
    alpha_vis = (alpha_np * 255).astype(np.uint8)
    alpha_vis_rgb = cv2.cvtColor(alpha_vis, cv2.COLOR_GRAY2RGB)

    return wm_np, alpha_vis_rgb


# -------------------------------------------------
# Watermark wrapper (supports custom image)
# -------------------------------------------------
def run_watermark(selected_image, custom_image, use_custom, corner, strength, progress=gr.Progress()):
    if wm_model is None:
        return None, None, "Model not loaded", "Model not loaded", None

    if use_custom and custom_image is not None:
        image_to_use = custom_image
    else:
        image_to_use = selected_image

    if image_to_use is None:
        return None, None, "No image selected", "No image selected", None

    progress(0.2, desc="Processing watermark...")
    original_np = np.array(image_to_use.convert("RGB"))
    wm_np, alpha_np = process_watermark_np(original_np, corner, strength)

    progress(0.8, desc="Computing quality metrics...")
    psnr_value, ssim_value = compute_psnr_ssim(original_np, wm_np)

    # Save watermarked image for download
    progress(0.9, desc="Preparing download...")
    wm_image = Image.fromarray(wm_np)
    temp_dir = tempfile.mkdtemp()
    download_path = os.path.join(temp_dir, "watermarked_image.png")
    wm_image.save(download_path)

    progress(1.0, desc="Done!")
    return (
        wm_image,
        Image.fromarray(alpha_np),
        f"{psnr_value:.4f} dB",
        f"{ssim_value:.4f}",
        download_path
    )


# -------------------------------------------------
# Gallery selection handler
# -------------------------------------------------
def pick_image(evt: gr.SelectData, images):
    return images[evt.index]


# -------------------------------------------------
# Gradio UI
# -------------------------------------------------
with gr.Blocks(
    title="PhotoMaker Extension - CAP-C6-Group-3",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # PhotoMaker Extension
        **CAP-C6-Group-3** | Multi-Identity Face Generation with Watermarking
        """
    )

    with gr.Tabs():

        # ---------------- PhotoMaker Tab ----------------
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="Input Image",
                        height=300
                    )
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="e.g., 'a photo of img1 and img2 standing together'",
                        info="Use img1, img2, etc. to reference detected faces"
                    )
                    run_btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    output_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=2,
                        height=350,
                        object_fit="contain",
                        show_label=True
                    )
                    with gr.Row():
                        gr.Markdown("*Click an image to select it for watermarking*")
                        download_generated = gr.File(
                            label="Download All",
                            visible=True,
                            file_count="single"
                        )

            with gr.Accordion("ArcFace Identity Similarity", open=True):
                similarity_box = gr.Textbox(
                    label="Face Matching Results",
                    lines=6,
                    interactive=False,
                    info="Cosine similarity between input and generated faces (0-1 scale)"
                )

            gallery_state = gr.State([])
            selected_image_state = gr.State()

            run_btn.click(
                fn=run_photomaker,
                inputs=[input_image, prompt],
                outputs=[output_gallery, similarity_box, gallery_state, download_generated]
            )

            output_gallery.select(
                fn=pick_image,
                inputs=[gallery_state],
                outputs=selected_image_state
            )

        # ---------------- Watermark Tab ----------------
        with gr.Tab("Watermark"):
            gr.Markdown("### Add Invisible Watermark")

            # Model loading section
            with gr.Accordion("Watermark Model Settings", open=(wm_model is None)):
                with gr.Row():
                    model_path_input = gr.Textbox(
                        label="Model Path",
                        placeholder="/path/to/watermark_model.pth",
                        value=wm_model_path or "",
                        scale=4
                    )
                    load_model_btn = gr.Button("Load Model", scale=1)

                model_status = gr.Textbox(
                    label="Status",
                    value="Model loaded" if wm_model is not None else "No model loaded",
                    interactive=False
                )

                def handle_load_model(path):
                    status, success = load_watermark_model(path)
                    return status

                load_model_btn.click(
                    fn=handle_load_model,
                    inputs=[model_path_input],
                    outputs=[model_status]
                )

            # Watermark controls
            with gr.Row():
                with gr.Column(scale=1):
                    selected_image_display = gr.Image(
                        type="pil",
                        label="Selected Image",
                        height=250
                    )

                    use_custom_toggle = gr.Checkbox(
                        label="Use custom image instead",
                        value=False
                    )

                    custom_image_upload = gr.Image(
                        type="pil",
                        label="Upload Custom Image",
                        visible=False,
                        height=200
                    )

                    use_custom_toggle.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=use_custom_toggle,
                        outputs=custom_image_upload
                    )

                with gr.Column(scale=1):
                    corner_dropdown = gr.Dropdown(
                        choices=CORNERS,
                        value="bottom-right",
                        label="Watermark Position"
                    )

                    strength_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Watermark Strength"
                    )

                    wm_btn = gr.Button("Apply Watermark", variant="primary")

                    with gr.Row():
                        psnr_box = gr.Textbox(
                            label="PSNR",
                            interactive=False,
                            scale=1
                        )
                        ssim_box = gr.Textbox(
                            label="SSIM",
                            interactive=False,
                            scale=1
                        )

            with gr.Row():
                wm_output = gr.Image(
                    type="pil",
                    label="Watermarked Output",
                    height=300
                )
                alpha_output = gr.Image(
                    type="pil",
                    label="Alpha Matte (Visualization)",
                    height=300
                )

            download_watermarked = gr.File(
                label="Download Watermarked Image",
                visible=True,
                file_count="single"
            )

            selected_image_state.change(
                fn=lambda img: img,
                inputs=selected_image_state,
                outputs=selected_image_display
            )

            wm_btn.click(
                fn=run_watermark,
                inputs=[
                    selected_image_state,
                    custom_image_upload,
                    use_custom_toggle,
                    corner_dropdown,
                    strength_slider
                ],
                outputs=[wm_output, alpha_output, psnr_box, ssim_box, download_watermarked]
            )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

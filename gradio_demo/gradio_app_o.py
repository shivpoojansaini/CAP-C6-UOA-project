import gradio as gr
from PIL import Image
import torch
import numpy as np
import cv2
import math
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
wm_model: WatermarkInjectionNetwork = load_model(MODEL_PATH, wm_device)


# -------------------------------------------------
# PhotoMaker wrapper
# -------------------------------------------------
def run_photomaker(image, prompt):
    import photomaker_cli

    if image is None:
        return [], "", []

    temp_path = "/tmp/pm_input.png"
    image.save(temp_path)

    photomaker_cli.INPUT_IMAGES = [temp_path]
    photomaker_cli.PROMPT = prompt
    photomaker_cli.NEGATIVE_PROMPT = (
        "nsfw, lowres, bad anatomy, bad hands, text, error"
    )

    images, seed = generate_image(pipe, face_detector, device)
    pil_images = list(images)

    return pil_images, "", pil_images


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
def run_watermark(selected_image, custom_image, use_custom, corner, strength):
    if use_custom and custom_image is not None:
        image_to_use = custom_image
    else:
        image_to_use = selected_image

    if image_to_use is None:
        return None, None, None, None

    original_np = np.array(image_to_use.convert("RGB"))
    wm_np, alpha_np = process_watermark_np(original_np, corner, strength)

    psnr_value, ssim_value = compute_psnr_ssim(original_np, wm_np)

    return (
        Image.fromarray(wm_np),
        Image.fromarray(alpha_np),
        f"{psnr_value:.4f} dB",
        f"{ssim_value:.4f}"
    )


# -------------------------------------------------
# Gallery selection handler
# -------------------------------------------------
def pick_image(evt: gr.SelectData, images):
    return images[evt.index]


# -------------------------------------------------
# Gradio UI
# -------------------------------------------------
with gr.Blocks(title="PhotoMaker Extension by CAP-C6-Group-3") as demo:

    gr.Markdown("## PhotoMaker Extension by CAP-C6-Group-3")

    with gr.Tabs():

        # ---------------- PhotoMaker Tab ----------------
        with gr.Tab("PhotoMaker"):

            input_image = gr.Image(type="pil", label="Upload Input Image")
            prompt = gr.Textbox(label="Prompt (must include img1, img2, ...)")

            run_btn = gr.Button("Generate Images")

            output_gallery = gr.Gallery(
                label="Generated Images (Click to Select)",
                interactive=True
            )

            identity_box = gr.Textbox(label="Identity Assignment", lines=4)

            gallery_state = gr.State([])
            selected_image_state = gr.State()

            run_btn.click(
                fn=run_photomaker,
                inputs=[input_image, prompt],
                outputs=[output_gallery, identity_box, gallery_state]
            )

            output_gallery.select(
                fn=pick_image,
                inputs=[gallery_state],
                outputs=selected_image_state
            )

        # ---------------- Watermark Tab ----------------
        with gr.Tab("Watermark"):

            gr.Markdown("### Single Image Watermarking")

            selected_image_display = gr.Image(
                type="pil",
                label="Selected Image from PhotoMaker"
            )

            use_custom_toggle = gr.Checkbox(
                label="Use a different image instead of PhotoMaker output?",
                value=False
            )

            custom_image_upload = gr.Image(
                type="pil",
                label="Upload Custom Image",
                visible=False
            )

            use_custom_toggle.change(
                fn=lambda x: gr.update(visible=x),
                inputs=use_custom_toggle,
                outputs=custom_image_upload
            )

            corner_dropdown = gr.Dropdown(
                choices=CORNERS,
                value="bottom-right",
                label="Watermark Corner"
            )

            strength_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Watermark Strength"
            )

            wm_btn = gr.Button("Add Watermark")

            with gr.Row():
                wm_output = gr.Image(type="pil", label="Watermarked Output")
                alpha_output = gr.Image(type="pil", label="Alpha Matte")

            with gr.Accordion("Quality Metrics (PSNR & SSIM)", open=False):
                psnr_box = gr.Textbox(label="PSNR", interactive=False)
                ssim_box = gr.Textbox(label="SSIM", interactive=False)

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
                outputs=[wm_output, alpha_output, psnr_box, ssim_box]
            )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

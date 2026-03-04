import numpy as np
import torch

def extract_unet_face_feature(
    hidden_states,      # [B, C, H_feat, W_feat]
    debug_image,        # PIL image (decoded at timestep t)
    face_detector,      # your InsightFace detector
    analyze_faces_fn,   # your analyze_faces function
):
    """
    Returns:
        List of face feature tensors [C] (one per detected face)
    """

    if hidden_states.dim() != 4:
        return []

    B, C, H_feat, W_feat = hidden_states.shape

    # --------------------------------------------------
    # 1️⃣ Detect faces in decoded image
    # --------------------------------------------------

    img_w, img_h = debug_image.size
    img_array = np.array(debug_image)[:, :, ::-1]  # RGB → BGR

    faces = analyze_faces_fn(face_detector, img_array)

    if len(faces) == 0:
        print("⚠️ No face detected in debug image.")
        return []

    # Sort left → right for consistency
    faces = sorted(faces, key=lambda f: f.bbox[0])

    face_features = []

    # --------------------------------------------------
    # 2️⃣ Convert image bbox → feature bbox
    # --------------------------------------------------

    for face in faces:

        x1, y1, x2, y2 = face.bbox  # pixel coordinates

        # Normalize to feature map resolution
        x1_feat = int(max(0, min(W_feat, x1 / img_w * W_feat)))
        x2_feat = int(max(0, min(W_feat, x2 / img_w * W_feat)))
        y1_feat = int(max(0, min(H_feat, y1 / img_h * H_feat)))
        y2_feat = int(max(0, min(H_feat, y2 / img_h * H_feat)))

        if x2_feat <= x1_feat or y2_feat <= y1_feat:
            continue

        # --------------------------------------------------
        # 3️⃣ Extract region from UNet feature map
        # --------------------------------------------------

        region = hidden_states[:, :, y1_feat:y2_feat, x1_feat:x2_feat]

        if region.numel() == 0:
            continue

        # --------------------------------------------------
        # 4️⃣ Global average pooling → face embedding
        # --------------------------------------------------

        face_feature = region.mean(dim=[2, 3])  # [B, C]

        # Remove batch dimension
        face_feature = face_feature[0].detach().cpu()

        face_features.append(face_feature)

    return face_features
import os
import pickle
import base64
import hashlib
import io
import numpy as np
import cv2
import PIL.Image
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# 1. Dashboard configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DFU Diagnostic & Benchmark Tool",
    page_icon="🩺",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Healthy", "DFU Detected"]
ANATOMICAL_REJECTION_MESSAGE = (
    "⛔ Anatomical Rejection: Hand or non-foot image detected. "
    "Please upload a valid lower-extremity foot image."
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# CNN models are more reliable on real photos than pixel-level Random Forest.
MODEL_WEIGHTS = {
    "Swin Transformer": 0.21,
    "EfficientNet-B0": 0.26,
    "DenseNet-121": 0.24,
    "MobileNetV2": 0.24,
    "Random Forest": 0.05,
}

SENSITIVITY_PRESETS = {
    "Balanced (recommended)": {
        "dfu_threshold": 0.50,
        "hot_patch": 0.72,
        "min_hot_patches": 2,
        "inflammation_boost": 0.08,
    },
    "High precision (fewer false DFU)": {
        "dfu_threshold": 0.58,
        "hot_patch": 0.80,
        "min_hot_patches": 2,
        "inflammation_boost": 0.04,
    },
    "High recall (catch hidden ulcers)": {
        "dfu_threshold": 0.40,
        "hot_patch": 0.62,
        "min_hot_patches": 1,
        "inflammation_boost": 0.12,
    },
}

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --primary-color: #0284c7;
        --medical-blue: #0284c7;
        --page-background: #f3f8fb;
        --card-background: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 5%, rgba(14, 165, 233, 0.09), transparent 28%),
            var(--page-background);
    }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
        border-right: 1px solid #dbeafe;
    }
    [data-testid="stHeader"] { background: rgba(243, 248, 251, 0.92); }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #164e63; font-weight: 600; }
    [data-testid="stSlider"] [role="slider"] {
        background: #0284c7 !important; border-color: #0284c7 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"] { background: #bae6fd; }
    div.stButton > button {
        border-radius: 10px; border-color: #bae6fd; color: #0369a1;
        font-weight: 650; transition: all 180ms ease;
    }
    div.stButton > button:hover {
        color: #ffffff; border-color: #0284c7; background: #0284c7;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.22);
    }
    .hero-shell {
        position: relative; display: flex; align-items: center; justify-content: space-between;
        gap: 2rem; min-height: 290px; overflow: hidden; padding: 2.3rem 2.6rem;
        margin: 0 0 1.4rem; border-radius: 28px; color: #f0fdfa;
        background: linear-gradient(118deg, #082f49 0%, #075985 56%, #0e7490 100%);
        box-shadow: 0 24px 55px rgba(8, 47, 73, 0.18);
    }
    .hero-shell::before, .hero-shell::after {
        content: ""; position: absolute; border-radius: 50%; pointer-events: none;
    }
    .hero-shell::before {
        width: 420px; height: 420px; right: 9%; top: -270px;
        border: 1px solid rgba(165, 243, 252, 0.28);
        box-shadow: 0 0 0 34px rgba(165, 243, 252, 0.045), 0 0 0 70px rgba(165, 243, 252, 0.035);
        animation: orbit-drift 12s ease-in-out infinite alternate;
    }
    .hero-shell::after {
        width: 220px; height: 220px; right: 20%; bottom: -170px;
        background: rgba(34, 211, 238, 0.12); filter: blur(8px);
    }
    .hero-copy, .hero-visual { position: relative; z-index: 1; }
    .hero-copy { max-width: 660px; }
    .hero-eyebrow {
        color: #a5f3fc; font-size: 0.76rem; font-weight: 750; letter-spacing: 0.19em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 0.8rem 0 0.65rem; color: #ffffff; font-size: clamp(2.1rem, 4vw, 3.5rem);
        font-weight: 760; letter-spacing: -0.045em; line-height: 1.03;
    }
    .hero-description { max-width: 560px; color: #dbeafe; font-size: 1rem; line-height: 1.7; }
    .hero-status {
        display: inline-flex; align-items: center; gap: 0.55rem; margin-top: 1.2rem;
        padding: 0.48rem 0.85rem; border: 1px solid rgba(165, 243, 252, 0.28);
        border-radius: 999px; background: rgba(8, 47, 73, 0.38);
        color: #ecfeff; font-size: 0.82rem; font-weight: 650;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #5eead4;
        box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.55); animation: pulse-health 2s infinite;
    }
    .hero-visual {
        display: grid; place-items: center; width: 210px; height: 210px; flex: 0 0 210px;
        position: relative; border: 1px solid rgba(207, 250, 254, 0.22); border-radius: 50%;
        background: radial-gradient(circle, rgba(103, 232, 249, 0.22), rgba(8, 47, 73, 0.08) 69%);
    }
    .hero-ring {
        position: absolute; width: 178px; height: 178px; border-radius: 50%;
        border: 1px dashed rgba(165, 243, 252, 0.7);
        animation: slow-spin 9s linear infinite;
    }
    .hero-icon {
        display: grid; place-items: center; width: 112px; height: 112px; overflow: hidden;
        position: relative; border: 1px solid rgba(255,255,255,0.22); border-radius: 32px;
        background: linear-gradient(145deg, rgba(255,255,255,0.2), rgba(255,255,255,0.07));
        color: #ecfeff; box-shadow: inset 0 1px 0 rgba(255,255,255,0.25);
        animation: float-medical 3.2s ease-in-out infinite;
    }
    .hero-icon svg { width: 74px; height: 74px; filter: drop-shadow(0 4px 9px rgba(34,211,238,0.24)); }
    .hero-scan {
        position: absolute; z-index: 2; left: 13px; right: 13px; top: 4px; height: 2px;
        background: #a5f3fc; box-shadow: 0 0 10px 3px rgba(103,232,249,0.76);
        animation: scan-hero 2.8s ease-in-out infinite alternate;
    }
    .hero-scan::before, .hero-scan::after {
        content: ""; position: absolute; top: -3px; width: 8px; height: 8px;
        border: 1px solid #cffafe; border-radius: 50%; background: #06b6d4;
    }
    .hero-scan::before { left: -3px; }
    .hero-scan::after { right: -3px; }
    .hero-crosshair {
        position: absolute; z-index: 3; width: 130px; height: 130px; border: 1px solid rgba(165,243,252,0.13);
        border-radius: 50%; pointer-events: none;
    }
    .step-track {
        display: flex; align-items: center; gap: 0.7rem; margin: 0.5rem 0 1.25rem;
        color: #64748b; font-size: 0.82rem; font-weight: 650;
    }
    .step-chip {
        display: inline-flex; align-items: center; gap: 0.48rem;
        padding: 0.55rem 0.8rem; border: 1px solid #dbeafe; border-radius: 999px;
        background: rgba(255,255,255,0.85);
    }
    .step-chip.active { border-color: #7dd3fc; background: #e0f2fe; color: #075985; }
    .step-number {
        display: grid; place-items: center; width: 1.35rem; height: 1.35rem;
        border-radius: 50%; background: #e2e8f0; color: #475569; font-size: 0.72rem;
    }
    .step-chip.active .step-number { background: #0284c7; color: white; }
    .step-connector { width: clamp(12px, 4vw, 52px); height: 1px; background: #cbd5e1; }
    .welcome-grid {
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem; margin: 1.2rem 0 1rem;
    }
    .welcome-card {
        min-height: 148px; padding: 1.15rem; border: 1px solid #e0f2fe;
        border-radius: 17px; background: rgba(255,255,255,0.88);
        box-shadow: 0 6px 18px rgba(8, 47, 73, 0.035);
        animation: rise-in 500ms ease both;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    .welcome-card:nth-child(2) { animation-delay: 90ms; }
    .welcome-card:nth-child(3) { animation-delay: 180ms; }
    .welcome-card:hover { transform: translateY(-4px); box-shadow: 0 14px 28px rgba(8, 47, 73, 0.09); }
    .welcome-icon {
        display: grid; place-items: center; width: 38px; height: 38px; margin-bottom: 0.8rem;
        border-radius: 12px; background: #e0f2fe; color: #0369a1; font-size: 1.2rem;
    }
    .welcome-title { color: #0f172a; font-weight: 700; margin-bottom: 0.25rem; }
    .welcome-copy { color: #64748b; font-size: 0.88rem; line-height: 1.5; }
    .upload-intro { text-align: left; color: #475569; }
    .upload-intro h3 { margin-bottom: 0.15rem; color: #0f172a; font-size: 1.25rem; }
    .upload-intro p { margin-bottom: 0.9rem; color: #64748b; font-size: 0.92rem; }
    [data-testid="stFileUploader"] {
        padding: 0.45rem; border: 1px solid #e0f2fe; border-radius: 18px;
        background: linear-gradient(135deg, #f0f9ff, #ffffff 58%, #f0fdfa);
        transition: box-shadow 180ms ease, transform 180ms ease;
    }
    [data-testid="stFileUploader"]:hover {
        box-shadow: 0 10px 30px rgba(14, 116, 144, 0.11); transform: translateY(-2px);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #7dd3fc; border-radius: 13px; background: rgba(255,255,255,0.88);
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--medical-blue); }
    @keyframes pulse-health {
        0%, 100% { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.52); }
        50% { box-shadow: 0 0 0 8px rgba(94, 234, 212, 0); }
    }
    @keyframes float-medical { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
    @keyframes slow-spin { to { transform: rotate(360deg); } }
    @keyframes scan-hero { from { transform: translateY(2px); opacity: 0.72; } to { transform: translateY(102px); opacity: 1; } }
    @keyframes orbit-drift { to { transform: translate(-18px, 14px) rotate(12deg); } }
    @keyframes rise-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-background); border-radius: 14px;
        transition: box-shadow 180ms ease, border-color 180ms ease, transform 180ms ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        transform: translateY(-2px);
    }
    .scanner-frame {
        position: relative; max-width: 520px; margin: 0 auto 1rem;
        overflow: hidden; border-radius: 18px; background: #ffffff;
        border: 1px solid #bae6fd; box-shadow: 0 12px 30px rgba(8, 47, 73, 0.1);
    }
    .scanner-frame img { display: block; width: 100%; max-height: 340px; object-fit: contain; }
    .scanner-frame::after {
        content: ""; position: absolute; left: 0; right: 0; top: -4px;
        height: 3px; background: #67e8f9;
        box-shadow: 0 0 14px 4px rgba(6, 182, 212, 0.55);
        animation: scan-image 2.7s ease-in-out infinite alternate;
    }
    @keyframes scan-image {
        from { transform: translateY(0); }
        to { transform: translateY(350px); }
    }
    @media (max-width: 700px) {
        .hero-shell { min-height: 245px; padding: 1.6rem; }
        .hero-visual { width: 100px; height: 100px; flex-basis: 100px; }
        .hero-ring { width: 84px; height: 84px; }
        .hero-icon { width: 58px; height: 58px; border-radius: 18px; }
        .hero-icon svg { width: 42px; height: 42px; }
        .hero-scan { left: 7px; right: 7px; }
        @keyframes scan-hero { from { transform: translateY(1px); opacity: 0.72; } to { transform: translateY(52px); opacity: 1; } }
        .step-track { flex-wrap: wrap; }
        .welcome-grid { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Grad-CAM
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("The selected layer did not produce Grad-CAM activations.")
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def overlay_heatmap(pil_img, heatmap, threshold=0.25):
    img = np.array(pil_img.resize((224, 224)))
    heatmap_masked = np.copy(heatmap)
    heatmap_masked[heatmap_masked < threshold] = 0.0
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_masked), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.65, heatmap_colored, 0.35, 0)


def skin_surface_heatmap(pil_img):
    """Create a localized visual focus map from texture variation and red-color cues."""
    rgb = np.array(pil_img.convert("RGB").resize((224, 224)))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    texture = np.abs(cv2.Laplacian(lightness, cv2.CV_32F, ksize=3))
    texture = cv2.GaussianBlur(texture, (11, 11), 0)

    low, high = np.percentile(texture, (20, 98))
    texture_map = np.clip((texture - low) / max(float(high - low), 1e-6), 0, 1)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 45, 45], np.uint8), np.array([12, 255, 255], np.uint8))
    red_high = cv2.inRange(hsv, np.array([168, 45, 45], np.uint8), np.array([180, 255, 255], np.uint8))
    erythema_map = cv2.GaussianBlur(cv2.bitwise_or(red_low, red_high).astype(np.float32) / 255.0, (15, 15), 0)

    focus = 0.65 * texture_map + 0.35 * erythema_map
    focus = cv2.GaussianBlur(focus, (7, 7), 0)
    maximum = float(np.max(focus))
    return focus / maximum if maximum > 0 else focus


# -----------------------------------------------------------------------------
# 3. Foot / skin isolation, lighting, inflammation cues
# -----------------------------------------------------------------------------
def _skin_mask(bgr):
    """Combine YCrCb + HSV masks so varied skin tones and indoor lighting still work."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_y = cv2.inRange(ycrcb, np.array([0, 125, 70], np.uint8), np.array([255, 180, 140], np.uint8))
    mask_h = cv2.inRange(hsv, np.array([0, 20, 40], np.uint8), np.array([30, 180, 255], np.uint8))
    mask_h2 = cv2.inRange(hsv, np.array([160, 20, 40], np.uint8), np.array([180, 180, 255], np.uint8))
    mask = cv2.bitwise_or(mask_y, cv2.bitwise_or(mask_h, mask_h2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def validate_foot_anatomy(image):
    """Reject only a clearly hand-shaped skin contour; ambiguous images pass for review."""
    rgb = np.array(image.convert("RGB"))
    if rgb.size == 0 or min(rgb.shape[:2]) < 64:
        return True

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = _skin_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    image_area = float(rgb.shape[0] * rgb.shape[1])
    if area / image_area < 0.015:
        return True

    _, _, width, height = cv2.boundingRect(contour)
    aspect_ratio = min(width, height) / max(width, height, 1)
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4 or area <= 0:
        return True

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return True

    diagonal = float(np.hypot(width, height))
    deep_gaps = sum(
        depth / 256.0 >= 0.025 * diagonal
        for depth in defects.reshape(-1, 4)[:, 3]
    )
    solidity = area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    hand_like_shape = aspect_ratio >= 0.42 and solidity < 0.93 and deep_gaps >= 4
    return not hand_like_shape


def isolate_foot(pil_img, enable_cropping=True):
    """Crop to the largest skin region and even out lighting (CLAHE)."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    notes = []

    mask = _skin_mask(bgr)
    coverage = float(np.mean(mask > 0))
    cropped = bgr
    used_crop = False

    if enable_cropping and coverage > 0.04:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1500:
                x, y, w, h = cv2.boundingRect(largest)
                pad_w, pad_h = int(w * 0.04), int(h * 0.04)
                y1, y2 = max(0, y - pad_h), min(bgr.shape[0], y + h + pad_h)
                x1, x2 = max(0, x - pad_w), min(bgr.shape[1], x + w + pad_w)
                cropped = bgr[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]
                used_crop = True
                notes.append("Foot/skin region isolated from background.")

    if coverage < 0.04:
        notes.append("Little skin detected. Use a closer, well-lit foot image.")

    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(l_ch)
    cropped = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    processed = PIL.Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return processed, mask, notes, used_crop


def tissue_cues(pil_img, skin_mask=None):
    """
    Detect inflammation / early ulcer signs the CNNs often miss:
    local erythema, yellow slough vs surrounding skin, dark necrosis.
    Scores are 0-1 relative to the rest of the foot, so normal skin does not saturate.
    """
    rgb = np.array(pil_img.convert("RGB"))
    if rgb.size == 0:
        return {"erythema": 0.0, "slough": 0.0, "necrosis": 0.0, "texture": 0.0, "risk": 0.0}

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if skin_mask is None or skin_mask.shape[:2] != gray.shape:
        skin_mask = np.full(gray.shape, 255, np.uint8)
    if skin_mask.max() == 0:
        skin_mask = np.full(gray.shape, 255, np.uint8)

    l_ch = lab[:, :, 0].astype(np.float32)
    a_ch = lab[:, :, 1].astype(np.float32) - 128.0
    b_ch = lab[:, :, 2].astype(np.float32) - 128.0

    valid = (skin_mask > 0) & (l_ch > 38)
    if not np.any(valid):
        return {"erythema": 0.0, "slough": 0.0, "necrosis": 0.0, "texture": 0.0, "risk": 0.0}

    med_a = float(np.median(a_ch[valid]))
    med_b = float(np.median(b_ch[valid]))
    med_l = float(np.median(l_ch[valid]))

    a_blur = cv2.GaussianBlur(a_ch, (21, 21), 0)
    local_red = np.clip(a_blur - med_a, 0, None)
    local_erythema = float(np.percentile(local_red[valid], 96))
    local_erythema = float(np.clip((local_erythema - 4.0) / 16.0, 0.0, 1.0))

    # Uniform redness across the whole foot (early cellulitis / inflammation)
    # without a focal ulcer. Use redness minus yellowness so brown skin is not flagged.
    red_index = a_ch - 0.45 * np.maximum(b_ch, 0)
    global_erythema = float(np.mean(red_index[valid]))
    global_erythema = float(np.clip((global_erythema - 6.0) / 14.0, 0.0, 1.0))
    erythema = float(max(local_erythema, global_erythema))

    yellow = (b_ch > (med_b + 14.0)) & (b_ch > 18.0) & (a_ch < (med_a + 4.0)) & valid
    slough = float(np.clip(float(np.mean(yellow)) / 0.08, 0.0, 1.0))

    dark = (l_ch < (med_l - 38.0)) & (l_ch < 55.0) & valid
    necrosis = float(np.clip(float(np.mean(dark)) / 0.08, 0.0, 1.0))

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = float(np.std(lap[valid]))
    texture = float(np.clip((texture - 18.0) / 30.0, 0.0, 1.0))

    risk = float(np.clip(0.55 * erythema + 0.30 * slough + 0.15 * necrosis, 0.0, 1.0))
    return {
        "erythema": erythema,
        "slough": slough,
        "necrosis": necrosis,
        "texture": texture,
        "risk": risk,
    }


def analysis_views(pil_img):
    """
    Models were trained on close-up tissue patches, not whole-foot photos.
    Scan the isolated foot at several crops so a small / hidden wound can still score high.
    """
    views = [("full", pil_img)]
    w, h = pil_img.size
    if w < 32 or h < 32:
        return views

    def add_crop(name, box):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 >= 32 and y2 - y1 >= 32:
            views.append((name, pil_img.crop((x1, y1, x2, y2))))

    pw, ph = max(w // 2, 32), max(h // 2, 32)
    add_crop("tl", (0, 0, pw, ph))
    add_crop("tr", (w - pw, 0, w, ph))
    add_crop("bl", (0, h - ph, pw, h))
    add_crop("br", (w - pw, h - ph, w, h))
    add_crop("center", (int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82)))

    # Three overlapping vertical strips help when the wound sits on the plantar surface.
    sw = max(int(w * 0.45), 32)
    add_crop("left", (0, 0, sw, h))
    add_crop("mid", ((w - sw) // 2, 0, (w + sw) // 2, h))
    add_crop("right", (w - sw, 0, w, h))
    return views


def to_tensor(pil_img):
    return TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)


# -----------------------------------------------------------------------------
# 4. Model loading
# -----------------------------------------------------------------------------
def find_model_file(keyword, models_dir="models"):
    if not os.path.exists(models_dir):
        return None
    for file in os.listdir(models_dir):
        if keyword in file and not file.endswith(".zip"):
            return os.path.join(models_dir, file)
    return None


def build_model_skeleton(arch_type):
    if arch_type == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
    elif arch_type == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif arch_type == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif arch_type == "swin":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, 2)
    else:
        raise ValueError(arch_type)
    return model


def resolve_class_order(checkpoint):
    """ImageFolder order is healthy=0, ulcer=1. Honour checkpoint mapping if present."""
    mapping = checkpoint.get("class_to_idx") if isinstance(checkpoint, dict) else None
    if not mapping:
        return 0, 1
    healthy_idx = mapping.get("healthy", mapping.get("Healthy", mapping.get("normal", 0)))
    ulcer_idx = mapping.get("ulcer", mapping.get("DFU", mapping.get("dfu", 1)))
    return int(healthy_idx), int(ulcer_idx)


@st.cache_resource
def load_all_models():
    loaded_models = {}
    class_order = (0, 1)
    model_configs = [
        ("MobileNetV2", "mobilenet", "mobilenet_v2_unified", 97.51),
        ("EfficientNet-B0", "efficientnet", "efficientnet_b0_unified", 97.75),
        ("DenseNet-121", "densenet", "densenet121_unified", 97.16),
        ("Swin Transformer", "swin", "swin_t_unified", 98.70),
    ]

    for display_name, arch_type, keyword, accuracy in model_configs:
        file_path = find_model_file(keyword)
        if not file_path:
            continue
        try:
            model = build_model_skeleton(arch_type)
            checkpoint = torch.load(file_path, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
                class_order = resolve_class_order(checkpoint)
            loaded_models[display_name] = ("pytorch", model, accuracy, arch_type)
        except Exception as e:
            st.sidebar.error(f"Failed loading {display_name}: {e}")

    rf_path = find_model_file("rf_unified")
    if rf_path:
        try:
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)
            loaded_models["Random Forest"] = ("sklearn", rf_model, 93.36, "rf")
        except Exception as e:
            st.sidebar.error(f"Failed loading Random Forest: {e}")

    return loaded_models, class_order


# -----------------------------------------------------------------------------
# 5. Inference
# -----------------------------------------------------------------------------
def predict_pytorch(model, pil_views, healthy_idx, ulcer_idx, use_tta=True):
    dfu_scores = []
    with torch.no_grad():
        for _, view in pil_views:
            variants = [view]
            if use_tta:
                variants.append(view.transpose(PIL.Image.FLIP_LEFT_RIGHT))
            probs_accum = []
            for img in variants:
                outputs = model(to_tensor(img))
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
                probs_accum.append(probs)
            mean_p = np.mean(probs_accum, axis=0)
            dfu_scores.append(float(mean_p[ulcer_idx]))
    return np.array(dfu_scores, dtype=np.float32)


def predict_rf(model, pil_views, healthy_idx, ulcer_idx):
    dfu_scores = []
    for _, view in pil_views:
        tensor = to_tensor(view)
        flat = nn.functional.interpolate(tensor, size=(64, 64)).cpu().numpy().reshape(1, -1)
        if hasattr(model, "n_features_in_") and flat.shape[1] != model.n_features_in_:
            needed = model.n_features_in_
            if flat.shape[1] > needed:
                flat = flat[:, :needed]
            else:
                pad = np.zeros((1, needed), dtype=flat.dtype)
                pad[:, :flat.shape[1]] = flat
                flat = pad
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flat)[0]
            dfu_scores.append(float(probs[ulcer_idx] if len(probs) > ulcer_idx else probs[-1]))
        else:
            pred = int(model.predict(flat)[0])
            dfu_scores.append(1.0 if pred == ulcer_idx else 0.0)
    return np.array(dfu_scores, dtype=np.float32)


def aggregate_patch_scores(scores, preset):
    """Blend mean + high percentile so one noisy crop cannot dominate."""
    if scores.size == 0:
        return 0.0, 0
    hot = int(np.sum(scores >= preset["hot_patch"]))
    blended = 0.55 * float(np.mean(scores)) + 0.45 * float(np.percentile(scores, 90))
    return float(np.clip(blended, 0.0, 1.0)), hot


def decide_verdict(cnn_scores, ensemble_dfu, cues, preset):
    """Require CNN consensus so one aggressive backbone cannot flip a healthy foot to DFU."""
    scores = np.array(cnn_scores, dtype=np.float32)
    n_pos = int(np.sum(scores >= preset["dfu_threshold"]))
    n_strong = int(np.sum(scores >= 0.70))

    lesion_cues = cues["slough"] >= 0.40 or cues["necrosis"] >= 0.35
    weak_visual = cues["slough"] < 0.18 and cues["necrosis"] < 0.18

    dfu_p = float(ensemble_dfu)
    if lesion_cues:
        dfu_p = min(1.0, dfu_p + 0.5 * preset["inflammation_boost"])

    strong_dfu = dfu_p >= preset["dfu_threshold"] and (
        n_strong >= 2
        or (
            n_pos >= 3
            and (lesion_cues or dfu_p >= preset["dfu_threshold"] + 0.08)
        )
    )
    if weak_visual:
        strong_dfu = n_strong >= 3 and dfu_p >= 0.72

    if strong_dfu:
        return "DFU Detected", dfu_p, "Open-wound / high-risk tissue pattern detected by multiple models."

    # Redness alone is not sufficient evidence for DFU: normal soles can be
    # red from pressure, lighting, temperature, or skin tone. Require CNN
    # support before presenting an early-risk warning.
    inflammation_supported = (
        cues["erythema"] >= 0.65
        and n_pos >= 2
        and dfu_p >= preset["dfu_threshold"] - 0.08
    )
    if inflammation_supported:
        return (
            "Possible inflammation",
            dfu_p,
            "No clear open ulcer, but redness on the foot suggests early DFU risk. Photograph the sole, heel and between the toes close-up.",
        )
    return "Healthy", dfu_p, "No consistent ulcer pattern across the CNN ensemble."


def run_benchmark(processed_pil, skin_mask, models_dict, class_order, preset, fast=False):
    healthy_idx, ulcer_idx = class_order
    views = analysis_views(processed_pil)
    if fast:
        views = views[:5]
        use_tta = False
    else:
        use_tta = True

    rows = []
    weighted_sum = 0.0
    weight_total = 0.0
    cam_model_item = None
    all_model_scores = {}
    cnn_dfu = []
    model_probabilities = {}

    for name, (m_type, model, bench_acc, arch_type) in models_dict.items():
        if m_type == "pytorch":
            scores = predict_pytorch(model, views, healthy_idx, ulcer_idx, use_tta=use_tta)
            if cam_model_item is None and arch_type in ("densenet", "efficientnet"):
                layer = model.features.denseblock4 if arch_type == "densenet" else model.features[-1]
                cam_model_item = (model, layer)
        else:
            scores = predict_rf(model, views[:3] if fast else views, healthy_idx, ulcer_idx)

        model_dfu, hot = aggregate_patch_scores(scores, preset)
        healthy_p = 1.0 - model_dfu
        pred_label = CLASS_NAMES[1] if model_dfu >= preset["dfu_threshold"] else CLASS_NAMES[0]
        model_weight = MODEL_WEIGHTS.get(name, 0.0)
        rows.append({
            "Model Architecture": name,
            "Prediction": pred_label,
            "Healthy Prob": f"{healthy_p * 100:.2f}%",
            "DFU Prob": f"{model_dfu * 100:.2f}%",
            "Ensemble weight": f"{model_weight * 100:.0f}%",
            "Hot patches": int(hot),
            "Benchmark Accuracy": f"{bench_acc:.2f}%",
        })
        all_model_scores[name] = scores
        model_probabilities[name] = model_dfu
        if model_weight:
            weighted_sum += model_weight * model_dfu
            weight_total += model_weight
        if m_type == "pytorch":
            cnn_dfu.append(model_dfu)

    ensemble_dfu = weighted_sum / weight_total if weight_total else 0.0
    cues = tissue_cues(processed_pil, skin_mask)
    verdict, adj_dfu, reason = decide_verdict(cnn_dfu, ensemble_dfu, cues, preset)
    return {
        "rows": rows,
        "ensemble_raw": ensemble_dfu,
        "ensemble_adj": adj_dfu,
        "verdict": verdict,
        "reason": reason,
        "cues": cues,
        "views": views,
        "cam_model_item": cam_model_item,
        "processed": processed_pil,
        "all_model_scores": all_model_scores,
        "model_probabilities": model_probabilities,
        "weight_total": weight_total,
        "ulcer_idx": ulcer_idx,
    }


def render_results(original_pil, processed_pil, notes, result, clinical_profile, numbness_areas):
    col1, col2 = st.columns([1, 1.25])
    with col1:
        with st.container(border=True):
            st.subheader("Image review")
            st.image(processed_pil, caption="Preprocessed image", use_container_width=True)
            with st.expander("Uploaded image"):
                st.image(original_pil, use_container_width=True)
            for note in notes:
                st.caption(note)

            cues = result["cues"]
            st.markdown("**Image-derived tissue cues**")
            cue_columns = st.columns(4)
            cue_columns[0].metric("Erythema", f"{cues['erythema'] * 100:.0f}%")
            cue_columns[1].metric("Slough cue", f"{cues['slough'] * 100:.0f}%")
            cue_columns[2].metric("Dark tissue cue", f"{cues['necrosis'] * 100:.0f}%")
            cue_columns[3].metric("Texture", f"{cues['texture'] * 100:.0f}%")

    with col2:
        with st.container(border=True):
            st.subheader("Ensemble screening result")
            verdict = result["verdict"]
            dfu_probability = result["ensemble_adj"]
            if (
                verdict == "DFU Detected"
                and not clinical_profile["diabetes_context"]
            ):
                verdict = "⚠️ Non-Diabetic Wound / Traumatic Injury Detected"
                st.warning(f"**Verdict:** {verdict}")
                st.caption(
                    "The image cannot determine the wound's cause or confirm diabetes status; "
                    "a clinical assessment is needed."
                )
            elif verdict == "Healthy":
                st.success(f"**Vision-model result:** No consistent ulcer pattern (score {dfu_probability * 100:.1f}%)")
            elif verdict == "Possible inflammation":
                st.warning(f"**Vision-model result:** Possible inflammation / early ulceration risk (score {dfu_probability * 100:.1f}%)")
            else:
                st.error(f"**Vision-model result:** DFU pattern detected (score {dfu_probability * 100:.1f}%)")
            st.caption(result["reason"])
            st.caption(
                f"Weighted ensemble before tissue adjustment: "
                f"{result['ensemble_raw'] * 100:.2f}%"
            )

            if clinical_profile["symptom_score"]:
                st.info(
                    f"Diabetes symptom score: {clinical_profile['symptom_score']}/2 "
                    f"({', '.join(clinical_profile['symptoms'])}). This is a screening prompt, not a diagnosis."
                )
            if clinical_profile["pregnant_postpartum"]:
                st.info(
                    "Pregnancy/postpartum status can change glucose risk. Consider gestational diabetes "
                    "screening with a clinician; the image screening threshold has been lowered."
                )
            if numbness_areas:
                st.warning(
                    "Neuropathy alert — reported loss of sensation is consistent with possible loss of "
                    "protective sensation (LOPS). Ulceration risk is elevated; arrange clinical foot assessment."
                )

        with st.container(border=True):
            st.subheader("Model contribution breakdown")
            probabilities = result["model_probabilities"]
            total_weight = result["weight_total"]
            available_names = [name for name in MODEL_WEIGHTS if name in probabilities]
            if available_names and total_weight > 0:
                weights = [MODEL_WEIGHTS[name] for name in available_names]
                scores = [probabilities[name] for name in available_names]
                contributions = [
                    weight / total_weight * score * 100
                    for weight, score in zip(weights, scores)
                ]
                figure = go.Figure(
                    go.Bar(
                        x=contributions,
                        y=available_names,
                        orientation="h",
                        marker_color="#0284c7",
                        customdata=np.column_stack(
                            (
                                np.array(weights) * 100,
                                np.array(scores) * 100,
                            )
                        ),
                        hovertemplate=(
                            "%{y}<br>Configured weight: %{customdata[0]:.0f}%"
                            "<br>Model DFU score: %{customdata[1]:.2f}%"
                            "<br>Weighted contribution: %{x:.2f} points<extra></extra>"
                        ),
                    )
                )
                figure.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=45),
                    xaxis_title="Weighted contribution to DFU score (percentage points)",
                    yaxis_title="",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    xaxis=dict(gridcolor="#e2e8f0", rangemode="tozero"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(figure, use_container_width=True)
                st.caption(
                    "Configured weights: Swin 21%, EfficientNet-B0 26%, DenseNet-121 24%, "
                    "MobileNetV2 24%, Random Forest 5%. Contributions are normalized if a model is unavailable."
                )
            else:
                st.info("Model contribution chart is unavailable because no weighted model predictions were produced.")

        with st.container(border=True):
            st.subheader("Per-model predictions")
            if result["rows"]:
                st.dataframe(result["rows"], use_container_width=True, hide_index=True)
            else:
                st.info("No models are available for inference.")

    with st.container(border=True):
        st.subheader("Skin surface focus map")
        st.image(
            overlay_heatmap(original_pil, skin_surface_heatmap(original_pil)),
            caption="Texture and red-color cues highlighted; this image-derived map does not confirm tissue damage.",
            width=420,
        )
        if result["cam_model_item"] is not None:
            st.markdown("**Model Grad-CAM**")
            py_model, target_layer = result["cam_model_item"]
            cam = GradCAM(py_model, target_layer)
            try:
                tensor = to_tensor(processed_pil).requires_grad_(True)
                cam_map = cam.generate_heatmap(tensor, class_idx=result["ulcer_idx"])
                st.image(
                    overlay_heatmap(processed_pil, cam_map),
                    caption="Model activation for the ulcer class; not a clinical measurement.",
                    width=420,
                )
            except (RuntimeError, ValueError, IndexError) as error:
                st.caption(f"Model Grad-CAM could not be generated: {error}")
            finally:
                cam.close()
                py_model.eval()


# -----------------------------------------------------------------------------
# 6. Clinical profile and upload-only Streamlit UI
# -----------------------------------------------------------------------------
st.markdown(
    """
    <section class="hero-shell">
      <div class="hero-copy">
        <div class="hero-eyebrow">DFU / VISION SCREENING</div>
        <div class="hero-title">A clearer view<br>of foot health.</div>
        <div class="hero-description">
          A thoughtful image-screening workspace combining visual model signals
          with patient-reported clinical context.
        </div>
        <div class="hero-status"><span class="status-dot"></span> Secure local image workflow · Upload only</div>
      </div>
      <div class="hero-visual" aria-hidden="true">
        <div class="hero-ring"></div>
        <div class="hero-crosshair"></div>
        <div class="hero-icon">
          <svg viewBox="0 0 100 110" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M29 29C22 25 20 18 14 18C8 18 7 25 11 31C15 37 21 41 22 48C24 59 14 68 16 81C18 95 31 102 50 102C69 102 82 95 84 81C86 68 76 59 78 48C79 41 85 37 89 31C93 25 92 18 86 18C80 18 78 25 71 29C64 33 59 34 50 34C41 34 36 33 29 29Z" fill="rgba(103,232,249,0.14)" stroke="#ecfeff" stroke-width="2.8" stroke-linejoin="round"/>
            <ellipse cx="14" cy="14" rx="5.5" ry="7" fill="#a5f3fc"/>
            <ellipse cx="27" cy="9" rx="5.7" ry="7.5" fill="#a5f3fc"/>
            <ellipse cx="42" cy="7" rx="6" ry="8" fill="#cfffff"/>
            <ellipse cx="58" cy="8" rx="5.7" ry="7.5" fill="#a5f3fc"/>
            <ellipse cx="73" cy="12" rx="5.3" ry="7" fill="#a5f3fc"/>
            <path d="M31 49C39 54 61 54 69 49M34 59C42 63 58 63 66 59" stroke="#67e8f9" stroke-width="2" stroke-linecap="round" opacity=".9"/>
            <path d="M36 81C43 77 57 77 64 81" stroke="#67e8f9" stroke-width="2" stroke-linecap="round" opacity=".75"/>
          </svg>
          <span class="hero-scan"></span>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Research screening prototype — not a medical device or diagnosis. "
    "Seek clinical assessment for concerning wounds, redness, pain, fever, or rapid change."
)

st.sidebar.header("Patient Clinical Profile")
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
water_intake = st.sidebar.slider("Daily water intake (L)", 0.5, 6.0, 2.0, 0.1)
urination_frequency = st.sidebar.slider(
    "Daily urination frequency (times/day)", 2, 15, 6
)
patient_weight = st.sidebar.number_input(
    "Patient weight (kg)", min_value=1.0, max_value=500.0, value=70.0, step=0.5
)
diabetes_history = st.sidebar.selectbox(
    "Known diabetes history", ["Type 1", "Type 2", "No/Unconfirmed"]
)
pregnant_postpartum = False
if gender == "Female":
    pregnant_postpartum = st.sidebar.checkbox("Pregnant / Postpartum Patient")
    if pregnant_postpartum:
        st.sidebar.info(
            "Gestational diabetes screening may be appropriate. Discuss testing with a clinician."
        )

symptoms = []
if water_intake > 4.0:
    symptoms.append("Polydipsia")
if urination_frequency > 8:
    symptoms.append("Polyuria")
symptom_score = len(symptoms)
if symptoms:
    st.sidebar.warning(
        f"Diabetes symptom score: {symptom_score}/2 ({', '.join(symptoms)}). "
        "Symptoms alone do not diagnose diabetes."
    )
else:
    st.sidebar.caption("Diabetes symptom score: 0/2")

numbness_areas = st.sidebar.multiselect(
    "Select Foot Areas with Numbness / Loss of Sensation",
    [
        "Toes/Digital",
        "Ball of Foot (Plantar)",
        "Arch",
        "Heel",
        "Ankle/Dorsum",
    ],
)
st.sidebar.divider()
preset_name = st.sidebar.selectbox(
    "Screening sensitivity", list(SENSITIVITY_PRESETS.keys()), index=0
)
preset = dict(SENSITIVITY_PRESETS[preset_name])
if pregnant_postpartum:
    preset["dfu_threshold"] = max(0.30, preset["dfu_threshold"] - 0.08)
enable_crop = st.sidebar.checkbox("Isolate skin region", value=True)
st.sidebar.caption(f"Active model threshold: {preset['dfu_threshold']:.2f}")

clinical_profile = {
    "symptoms": symptoms,
    "symptom_score": symptom_score,
    "pregnant_postpartum": pregnant_postpartum,
    "diabetes_context": (
        diabetes_history in ("Type 1", "Type 2")
        or symptom_score > 0
        or pregnant_postpartum
    ),
}

st.markdown(
    """
    <div class="step-track">
      <span class="step-chip active"><span class="step-number">1</span> Patient context</span>
      <span class="step-connector"></span>
      <span class="step-chip active"><span class="step-number">2</span> Foot image</span>
      <span class="step-connector"></span>
      <span class="step-chip"><span class="step-number">3</span> Screening report</span>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column = st.columns([0.7, 2.6, 0.7])[1]
with upload_column:
    with st.container(border=True):
        st.markdown(
            '<div class="upload-intro"><h3>Start with a foot photo</h3>'
            '<p>Drop an image here or browse your device. JPG, JPEG, and PNG · No camera stream.</p></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drag and drop or browse for an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )


def render_scanner_preview(image):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    st.markdown(
        f'<div class="scanner-frame"><img alt="Image being analyzed" '
        f'src="data:image/png;base64,{encoded_image}"></div>',
        unsafe_allow_html=True,
    )


if uploaded_file is None:
    st.markdown(
        """
        <div class="welcome-grid">
          <div class="welcome-card">
            <div class="welcome-icon">✦</div>
            <div class="welcome-title">A focused image check</div>
            <div class="welcome-copy">Image patterns are reviewed alongside model consensus to support, not replace, clinical judgment.</div>
          </div>
          <div class="welcome-card">
            <div class="welcome-icon">⌁</div>
            <div class="welcome-title">Your context matters</div>
            <div class="welcome-copy">Add symptom and sensation details in the profile to surface relevant screening prompts.</div>
          </div>
          <div class="welcome-card">
            <div class="welcome-icon">♡</div>
            <div class="welcome-title">Designed for a careful review</div>
            <div class="welcome-copy">Use an evenly lit close-up. For urgent or changing wounds, contact a healthcare professional.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    uploaded_bytes = uploaded_file.getvalue()
    image_signature = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
    anatomy_confirmation_key = f"confirm_foot_image_{image_signature}"
    try:
        original_pil = PIL.Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except (PIL.UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"The uploaded file could not be read as an image: {error}")
    else:
        anatomy_valid = validate_foot_anatomy(original_pil)
        if not anatomy_valid:
            st.error(ANATOMICAL_REJECTION_MESSAGE)
            st.caption(
                "This color-and-shape check is approximate and can mistake toes, cropped views, "
                "or unusual lighting for a hand."
            )
            if st.button(
                "This is a foot image — continue",
                key=f"confirm_foot_button_{image_signature}",
            ):
                st.session_state[anatomy_confirmation_key] = True
                st.rerun()

        if anatomy_valid or st.session_state.get(anatomy_confirmation_key, False):
            if not anatomy_valid:
                st.info("Foot image confirmed by the user; continuing to image screening.")
            render_scanner_preview(original_pil)
            if pregnant_postpartum:
                st.info(
                    "Pregnancy/postpartum status can change glucose risk. Consider gestational diabetes "
                    "screening with a clinician; the metabolic screening threshold is lowered."
                )
            with st.spinner("Analyzing the uploaded foot image..."):
                models_dict, class_order = load_all_models()
                if not models_dict:
                    st.error("No trained models could be loaded from the models/ folder.")
                else:
                    processed_pil, skin_mask, notes, _ = isolate_foot(
                        original_pil, enable_cropping=enable_crop
                    )
                    result = run_benchmark(
                        processed_pil,
                        skin_mask,
                        models_dict,
                        class_order,
                        preset,
                    )
                    render_results(
                        original_pil,
                        processed_pil,
                        notes,
                        result,
                        clinical_profile,
                        numbness_areas,
                    )

st.caption(
    f"Clinical profile recorded for this session: {gender}; "
    f"{patient_weight:.1f} kg; diabetes history: {diabetes_history}. "
    "This information supports screening context only."
)

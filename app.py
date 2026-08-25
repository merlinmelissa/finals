"""
Breast Cancer Mammogram Classifier — VGG16 + CLAHE (Final Model)
Streamlit version, for free deployment on Streamlit Community Cloud.
Educational/research use only — NOT a diagnostic tool.
"""

import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# CONFIG — update HF_MODEL_REPO to your own model repo path
# ---------------------------------------------------------------------------
HF_MODEL_REPO = "meli143/vgg16-mammogram-clahe"   
HF_MODEL_FILE = "vgg16_clahe_final.keras"
THRESHOLD = 0.5712
IMG_SIZE = 224
TEST_AUC = 0.751


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILE)
    return tf.keras.models.load_model(model_path)


model = load_model()


# ---------------------------------------------------------------------------
# Preprocessing — matches the training notebook exactly:
# crop_to_content -> CLAHE (native res) -> resize_with_pad -> /255
# ---------------------------------------------------------------------------
def crop_to_content(image_np, threshold=10):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return image_np
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image_np[rmin:rmax + 1, cmin:cmax + 1]


def apply_clahe(image_uint8):
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_for_model(pil_image, img_size=IMG_SIZE):
    img = np.array(pil_image.convert("RGB")).astype(np.uint8)
    img = crop_to_content(img)
    img = apply_clahe(img)
    img_resized = tf.image.resize_with_pad(
        tf.constant(img, dtype=tf.float32), img_size, img_size
    ).numpy()
    return (img_resized / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def make_gradcam(img_array, model):
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.cast(img_array, tf.float32), training=False)
        tape.watch(conv_out)
        loss = tf.reduce_sum(preds[:, 0])

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        grads = tf.ones_like(conv_out)

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def make_overlay(img_scaled, heatmap, img_size=IMG_SIZE):
    hm_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_uint8 = np.uint8(255 * np.clip(hm_resized, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(colored_rgb * 0.4 + img_scaled * 0.6, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Mammogram Classifier — VGG16 + CLAHE", layout="wide")

st.title("Breast Cancer Mammogram Classifier")
st.markdown(
    f"**Model:** VGG16 (ImageNet transfer learning) + CLAHE preprocessing &nbsp;|&nbsp; "
    f"**Test AUC:** {TEST_AUC} &nbsp;|&nbsp; **Threshold:** {THRESHOLD}"
)
st.warning(
    "This is a Final Year Project research/educational demo. "
    "It is **not a diagnostic tool** and must not be used for real medical decisions."
)

uploaded_file = st.file_uploader("Upload a mammogram image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    img_scaled = preprocess_for_model(pil_img)
    arr = np.expand_dims(img_scaled, 0).astype(np.float32)

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Malignant" if prob >= THRESHOLD else "Benign"

    heatmap = make_gradcam(arr, model)
    overlay = make_overlay(img_scaled, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM Overlay (model's focus region)", use_container_width=True)

    st.subheader(f"Prediction: {label}")
    st.write(f"Malignant probability: **{prob:.3f}** (decision threshold: {THRESHOLD})")
    st.caption(
        "Grad-CAM highlights the region the model weighted most heavily. "
        "Note: VGG16's final feature map is only 7×7, so localisation is coarse, not pixel-precise."
    )

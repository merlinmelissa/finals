"""
Mammogram classifier - ResNet50 + CLAHE
FYP research prototype, not for real diagnosis.
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

HF_REPO_ID = "meli143/resnet50-mammogram-clahe"
HF_FILENAME = "resnet50_clahe_final.keras"
IMG_SIZE = 224
THRESHOLD = 0.4721  # from find_best_threshold() on the val set, main run
TEST_AUC = 0.7447
MULTISEED_AUC = "0.7534 ± 0.0050"  # 3-seed robustness check, not a sig test

SAMPLES = {
    "Sample 1":  ("samples/benign_1.jpg",    "Benign"),
    "Sample 2":  ("samples/benign_2.jpg",    "Benign"),
    "Sample 3":  ("samples/benign_3.jpg",    "Benign"),
    "Sample 4":  ("samples/benign_4.jpg",    "Benign"),
    "Sample 5":  ("samples/benign_5.jpg",    "Benign"),
    "Sample 6":  ("samples/malignant_1.jpg", "Malignant"),
    "Sample 7":  ("samples/malignant_2.jpg", "Malignant"),
    "Sample 8":  ("samples/malignant_3.jpg", "Malignant"),
    "Sample 9":  ("samples/malignant_4.jpg", "Malignant"),
    "Sample 10": ("samples/malignant_5.jpg", "Malignant"),
}

st.set_page_config(page_title="Mammography Screening Assistant", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return tf.keras.models.load_model(path)


def crop_to_content(img, threshold=10):
    """Trim the black margins mammogram scans usually have."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax + 1, cmin:cmax + 1]


def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)


def preprocess_image(pil_image):
    # crop -> clahe -> resize -> resnet preprocessing (matches the notebook pipeline)
    img = np.array(pil_image.convert("RGB")).astype(np.uint8)
    img = apply_clahe(crop_to_content(img))
    resized = tf.image.resize_with_pad(tf.cast(img, tf.float32), IMG_SIZE, IMG_SIZE)

    display_img = resized.numpy() / 255.0
    model_input = tf.keras.applications.resnet50.preprocess_input(resized)
    return np.expand_dims(model_input.numpy(), axis=0), display_img


def make_gradcam(arr, model, threshold):
    layer = model.get_layer("conv4_block6_out")
    grad_model = tf.keras.Model(model.input, [layer.output, model.output])

    with tf.GradientTape() as tape:
        conv, pred = grad_model(arr, training=False)
        loss = tf.where(pred[:, 0] >= threshold, pred[:, 0], 1 - pred[:, 0])

    grads = tape.gradient(loss, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv[0] * weights[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def plt_colormap(heatmap):
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return coloured.astype(np.float32) / 255.0


def make_overlay(display_img, heatmap):
    # mask out background so the heatmap doesn't glow outside the breast tissue
    mask = (np.mean(display_img, axis=-1) > 0.03).astype(float)
    heatmap = heatmap * mask
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    coloured = plt_colormap(heatmap)
    return np.clip(display_img * 0.65 + coloured * 0.35, 0, 1)


st.title("🩺 Deep Learning Mammography Screening Assistant")
st.caption("Final Year Project — CBIS-DDSM mass classification using ResNet50 + CLAHE")
st.warning(
    "This application is a university research prototype. It has not been "
    "clinically validated and must not be used for real medical diagnosis "
    "or treatment decisions."
)

with st.expander("About this model"):
    st.write("**Architecture:** ResNet50 (ImageNet pretrained) + CLAHE contrast enhancement")
    st.write("**Task:** Binary classification of mammograms as benign or malignant")
    st.write(f"**Main test-set AUC:** {TEST_AUC:.4f}")
    st.write(
        "**Main test-set performance:** Accuracy 0.6190 | Sensitivity 0.8503 | "
        "Specificity 0.4719 | Precision 0.5061"
    )
    st.write(f"**3-seed mean AUC:** {MULTISEED_AUC}")
    st.caption("The three-seed run is a robustness check, not a formal significance test.")

with st.spinner("Loading ResNet50 + CLAHE model..."):
    model = load_model()

# ---- image source: upload or pick a pre-loaded sample ----
source = st.radio("Image source", ["Upload your own", "Try a sample"], horizontal=True)

pil_img = None
ground_truth = None

if source == "Upload your own":
    uploaded_file = st.file_uploader("Upload a mammogram image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
else:
    choice = st.selectbox("Pick a sample", list(SAMPLES.keys()))
    path, ground_truth = SAMPLES[choice]
    pil_img = Image.open(path)

if pil_img is not None:
    with st.spinner("Analysing mammogram..."):
        arr, display_img = preprocess_image(pil_img)
        prob = float(model.predict(arr, verbose=0)[0][0])
        label = "Malignant" if prob >= THRESHOLD else "Benign"
        heatmap = make_gradcam(arr, model, THRESHOLD)
        overlay = make_overlay(display_img, heatmap)

    col1, col2 = st.columns(2)
    col1.image(pil_img, caption="Input Mammogram", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    st.subheader("Prediction")
    if label == "Malignant":
        st.error(f"Model prediction: **{label}**")
    else:
        st.success(f"Model prediction: **{label}**")

    if ground_truth is not None:
        st.write(f"Ground truth for this sample: **{ground_truth}**")
        if ground_truth == label:
            st.caption("✅ Matches the model's prediction.")
        else:
            st.caption("⚠️ Doesn't match — a good example of where the model still gets it wrong.")

    metric1, metric2 = st.columns(2)
    metric1.metric("Malignant probability", f"{prob:.1%}")
    metric2.metric("Decision threshold", f"{THRESHOLD:.4f}")

    st.info(
        "Grad-CAM highlights image regions that contributed more strongly to "
        "the model's classification decision. It should not be interpreted "
        "as tumour segmentation or an exact lesion boundary."
    )

st.divider()
st.caption("Final Year Project research prototype | Dataset: CBIS-DDSM mass subset | Final model: ResNet50 + CLAHE")

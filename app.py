"""
Breast Cancer Mammogram Classifier — ResNet50 + CLAHE

Final Year Project research prototype.
Educational/research use only — NOT a diagnostic tool.
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIG
# ============================================================

HF_REPO_ID = "meli143/resnet50-mammogram-clahe"
HF_FILENAME = "resnet50_clahe_final.keras"

IMG_SIZE = 224

# Validation-selected threshold from main ResNet50 + CLAHE run
THRESHOLD = 0.4721

# Held-out test result from main single run
TEST_AUC = 0.7447

# 3-seed robustness result
MULTISEED_AUC = "0.7534 ± 0.0050"


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Mammography Screening Assistant",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME
    )

    return tf.keras.models.load_model(model_path)


# ============================================================
# PREPROCESSING
# ============================================================

def crop_to_content(image_np, threshold=10):
    # Remove black mammogram margins
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return image_np

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return image_np[
        rmin:rmax + 1,
        cmin:cmax + 1
    ]


def apply_clahe(image_uint8):
    # Same CLAHE settings as notebook
    gray = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )


def preprocess_image(pil_image):
    # Match final notebook pipeline:
    # crop -> CLAHE -> resize -> ResNet50 preprocessing

    img = np.array(
        pil_image.convert("RGB")
    ).astype(np.uint8)

    img = crop_to_content(img)
    img = apply_clahe(img)

    resized = tf.image.resize_with_pad(
        tf.cast(img, tf.float32),
        IMG_SIZE,
        IMG_SIZE
    )

    display_img = resized.numpy() / 255.0

    model_img = tf.keras.applications.resnet50.preprocess_input(
        resized
    )

    arr = np.expand_dims(
        model_img.numpy(),
        axis=0
    )

    return arr, display_img


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(arr, model, threshold):
    # Intermediate ResNet layer used in final notebook
    layer = model.get_layer(
        "conv4_block6_out"
    )

    grad_model = tf.keras.Model(
        model.input,
        [layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, pred = grad_model(
            arr,
            training=False
        )

        # Explain the predicted class
        loss = tf.where(
            pred[:, 0] >= threshold,
            pred[:, 0],
            1 - pred[:, 0]
        )

    grads = tape.gradient(
        loss,
        conv
    )

    weights = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        conv[0] * weights[0],
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = cv2.resize(
        heatmap.numpy(),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    return heatmap


def make_overlay(display_img, heatmap):
    # Keep Grad-CAM inside breast region
    mask = (
        np.mean(display_img, axis=-1) > 0.03
    ).astype(float)

    heatmap = heatmap * mask

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    coloured = plt_colormap(
        heatmap
    )

    overlay = np.clip(
        display_img * 0.65 +
        coloured * 0.35,
        0,
        1
    )

    return overlay


def plt_colormap(heatmap):
    # OpenCV JET colour map
    heatmap_uint8 = np.uint8(
        255 * np.clip(heatmap, 0, 1)
    )

    coloured = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    coloured = cv2.cvtColor(
        coloured,
        cv2.COLOR_BGR2RGB
    )

    return coloured.astype(
        np.float32
    ) / 255.0


# ============================================================
# UI
# ============================================================

st.title(
    "🩺 Deep Learning Mammography Screening Assistant"
)

st.caption(
    "Final Year Project — CBIS-DDSM mass classification using ResNet50 + CLAHE"
)

st.warning(
    "This application is a university research prototype. "
    "It has not been clinically validated and must not be used "
    "for real medical diagnosis or treatment decisions."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About this model"):

    st.write(
        "**Architecture:** ResNet50 "
        "(ImageNet pretrained) + CLAHE contrast enhancement"
    )

    st.write(
        "**Task:** Binary classification of mammograms "
        "as benign or malignant"
    )

    st.write(
        f"**Main test-set AUC:** {TEST_AUC:.4f}"
    )

    st.write(
        "**Main test-set performance:** "
        "Accuracy 0.6190 | "
        "Sensitivity 0.8503 | "
        "Specificity 0.4719 | "
        "Precision 0.5061"
    )

    st.write(
        f"**3-seed mean AUC:** {MULTISEED_AUC}"
    )

    st.caption(
        "The three-seed experiment is reported as a robustness "
        "check rather than a formal statistical significance test."
    )


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "Loading ResNet50 + CLAHE model..."
):
    model = load_model()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a mammogram image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    pil_img = Image.open(
        uploaded_file
    )

    with st.spinner(
        "Analysing mammogram..."
    ):

        arr, display_img = preprocess_image(
            pil_img
        )

        prob = float(
            model.predict(
                arr,
                verbose=0
            )[0][0]
        )

        label = (
            "Malignant"
            if prob >= THRESHOLD
            else "Benign"
        )

        heatmap = make_gradcam(
            arr,
            model,
            THRESHOLD
        )

        overlay = make_overlay(
            display_img,
            heatmap
        )


    # ========================================================
    # IMAGE RESULTS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            pil_img,
            caption="Uploaded Mammogram",
            use_container_width=True
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )


    # ========================================================
    # CLASSIFICATION RESULT
    # ========================================================

    st.subheader(
        "Prediction"
    )

    if label == "Malignant":

        st.error(
            f"Model prediction: **{label}**"
        )

    else:

        st.success(
            f"Model prediction: **{label}**"
        )


    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Malignant probability",
            f"{prob:.1%}"
        )

    with metric2:

        st.metric(
            "Decision threshold",
            f"{THRESHOLD:.4f}"
        )


    # ========================================================
    # INTERPRETABILITY NOTE
    # ========================================================

    st.info(
        "Grad-CAM highlights image regions that contributed more "
        "strongly to the model's classification decision. "
        "It should not be interpreted as tumour segmentation "
        "or an exact lesion boundary."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Final Year Project research prototype | "
    "Dataset: CBIS-DDSM mass subset | "
    "Final model: ResNet50 + CLAHE"
)
"""
Breast Cancer Mammogram Classifier — ResNet50 + CLAHE

Final Year Project research prototype.
Educational/research use only — NOT a diagnostic tool.
"""

import os
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
THRESHOLD = 0.4721

TEST_AUC = 0.7447
MULTISEED_AUC = "0.7534 ± 0.0050"


# ============================================================
# SAMPLE IMAGES
# ============================================================

SAMPLES = {
    "Sample 1": {
        "path": "samples/benign_1.jpg",
        "ground_truth": "Benign"
    },
    "Sample 2": {
        "path": "samples/benign_2.jpg",
        "ground_truth": "Benign"
    },
    "Sample 3": {
        "path": "samples/benign_3.jpg",
        "ground_truth": "Benign"
    },
    "Sample 4": {
        "path": "samples/benign_4.jpg",
        "ground_truth": "Benign"
    },
    "Sample 5": {
        "path": "samples/malignant_13.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 6": {
        "path": "samples/malignant_21.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 7": {
        "path": "samples/malignant_31.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 8": {
        "path": "samples/malignant_32.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 9": {
        "path": "samples/malignant_24.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 10": {
        "path": "samples/malignant_25.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 11": {
        "path": "samples/malignant_33.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 12": {
        "path": "samples/malignant_27.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 13": {
        "path": "samples/malignant_34.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 14": {
        "path": "samples/malignant_35.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 15": {
        "path": "samples/malignant_30.jpg",
        "ground_truth": "Malignant"
    },
    "Sample 16": {
        "path": "samples/benign_7.jpg",
        "ground_truth": "Benign"
    },
    "Sample 17": {
        "path": "samples/benign_8.jpg",
        "ground_truth": "Benign"
    },
    "Sample 18": {
        "path": "samples/benign_14.jpg",
        "ground_truth": "Benign"
    },
    "Sample 19": {
        "path": "samples/benign_10.jpg",
        "ground_truth": "Benign"
    }
}


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
    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

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
    # Same CLAHE settings used in the notebook
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
    # Final notebook pipeline:
    # crop -> CLAHE -> resize_with_pad -> ResNet50 preprocessing

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

    # Used only for displaying the processed image
    display_img = resized.numpy() / 255.0

    # Correct ResNet50 ImageNet preprocessing
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
    # Intermediate ResNet50 layer used in notebook
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

        # Explain the class actually predicted by the model
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
    # Keep Grad-CAM mostly within breast tissue
    mask = (
        np.mean(display_img, axis=-1) > 0.03
    ).astype(float)

    heatmap = heatmap * mask

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

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

    coloured = coloured.astype(
        np.float32
    ) / 255.0

    overlay = np.clip(
        display_img * 0.65 +
        coloured * 0.35,
        0,
        1
    )

    return overlay


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def run_prediction(pil_image, model):
    arr, display_img = preprocess_image(
        pil_image
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

    return label, prob, display_img, overlay


# ============================================================
# HEADER
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
# ABOUT MODEL
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
        "The three-seed analysis is reported as a robustness "
        "check rather than a formal statistical significance test."
    )


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "Loading ResNet50 + CLAHE model..."
):
    model = load_model()


st.success(
    "Model loaded successfully."
)


# ============================================================
# CHOOSE INPUT METHOD
# ============================================================

st.subheader(
    "Choose an image"
)

input_method = st.radio(
    "Select how you would like to test the model:",
    [
        "Use a sample image",
        "Upload your own image"
    ],
    horizontal=True
)


selected_image = None
ground_truth = None
selected_name = None


# ============================================================
# SAMPLE IMAGE OPTION
# ============================================================

if input_method == "Use a sample image":

    st.write(
        "Choose one of the six demonstration mammograms."
    )

    sample_name = st.selectbox(
        "Select sample",
        list(SAMPLES.keys())
    )

    sample_info = SAMPLES[
        sample_name
    ]

    sample_path = sample_info[
        "path"
    ]

    if os.path.exists(
        sample_path
    ):

        selected_image = Image.open(
            sample_path
        )

        ground_truth = sample_info[
            "ground_truth"
        ]

        selected_name = sample_name

    else:

        st.error(
            f"Sample file not found: {sample_path}"
        )


# ============================================================
# UPLOAD OPTION
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a mammogram image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        selected_image = Image.open(
            uploaded_file
        )

        selected_name = uploaded_file.name


# ============================================================
# ANALYSIS
# ============================================================

if selected_image is not None:

    st.divider()

    st.subheader(
        "Selected mammogram"
    )

    st.image(
        selected_image,
        caption=selected_name,
        width=350
    )


    if st.button(
        "Analyse Mammogram",
        type="primary"
    ):

        with st.spinner(
            "Running inference and generating Grad-CAM..."
        ):

            label, prob, display_img, overlay = run_prediction(
                selected_image,
                model
            )


        # ====================================================
        # RESULT
        # ====================================================

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


        metric1, metric2 = st.columns(
            2
        )

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


        # ====================================================
        # GROUND TRUTH FOR SAMPLE IMAGES
        # ====================================================

        if ground_truth is not None:

            st.write(
                f"**Known ground truth:** {ground_truth}"
            )

            if label == ground_truth:

                st.success(
                    "Prediction matches the known sample label."
                )

            else:

                st.warning(
                    "Prediction does not match the known sample label. "
                    "This example demonstrates a model error."
                )


        # ====================================================
        # VISUALISATIONS
        # ====================================================

        st.subheader(
            "Visual explanation"
        )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.image(
                selected_image,
                caption="Original Mammogram",
                use_container_width=True
            )

        with col2:

            st.image(
                display_img,
                caption="CLAHE-Enhanced Input",
                use_container_width=True
            )

        with col3:

            st.image(
                overlay,
                caption="Grad-CAM Overlay",
                use_container_width=True
            )


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

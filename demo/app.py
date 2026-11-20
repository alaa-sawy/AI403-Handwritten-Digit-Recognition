"""NeuroMotion: frozen local-checkpoint handwriting recognition demo."""

import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch

from src.inference import MODEL_CHECKPOINTS, checkpoint_for, load_checkpoint_model, predict_image

st.set_page_config(page_title="NeuroMotion | Handwritten Digits", page_icon="🔢", layout="centered")
st.title("🔢 NeuroMotion — Handwritten Digit-String Recognition")
st.caption("AI403 Final Project | ORAND-CAR-2014 | Local checkpoints + CTC")
st.info("Upload a cropped handwritten digit-string image. Predictions are text: leading zeros are preserved.")

model_label = st.selectbox("Choose model architecture", list(MODEL_CHECKPOINTS))
seed = st.selectbox("Checkpoint seed", [7, 21, 42], index=0)
model_name, _ = MODEL_CHECKPOINTS[model_label]
checkpoint_path = checkpoint_for(model_label, seed)
checkpoint_available = checkpoint_path.is_file()

if model_label == "Baseline CNN":
    st.warning(
        "Baseline is for scientific comparison, not reliable recognition: "
        "validation CER ≈ 78.36% and exact-match accuracy 0%."
    )
if checkpoint_available:
    st.caption(f"Local checkpoint ready: {checkpoint_path}")
else:
    st.warning(
        "Checkpoint missing locally for this selection. Copy the correct final best.pt "
        "from the team's Drive into: " + str(checkpoint_path)
    )

uploaded = st.file_uploader("Upload handwritten digit-string image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Cannot read uploaded image: {exc}")
    else:
        st.image(image, caption="Uploaded input", use_container_width=True)
        if st.button("Recognize digit string", type="primary", disabled=not checkpoint_available):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with st.spinner(f"Loading {model_label} checkpoint and running inference..."):
                    model, checkpoint, actual_device = load_checkpoint_model(
                        model_name, checkpoint_path, device, model_label=model_label
                    )
                    prediction, _, width = predict_image(image, model, actual_device)
                st.subheader("Recognition result")
                if prediction:
                    st.metric("Predicted digit string", prediction)
                else:
                    st.warning("CTC returned an empty digit string. Try a clearer cropped image.")
                st.caption(
                    f"Model: {model_label} | Seed: {checkpoint['config']['seed']} | "
                    f"Device: {actual_device} | Resized width: {width}px"
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
else:
    st.caption("Choose a model, verify its checkpoint is available, then upload an image.")

with st.expander("Model and evaluation notes"):
    st.markdown("""
    - This application **does not train** models or download pretrained weights.
    - Preprocessing: grayscale, fixed height 48 (aspect ratio preserved), normalize to [0, 1].
    - CTC collapses repeated frame predictions and removes blank tokens; repeated digits may remain when separated by blanks.
    - The demo is not an official TEST benchmark. See the project README for frozen TEST metrics.
    - Large final checkpoints are shared separately via the team's Drive, not automatically bundled with GitHub.
    """)

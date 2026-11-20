from pathlib import Path
import streamlit as st
from PIL import Image
import torch

from src.inference import MODEL_CHECKPOINTS, checkpoint_for, load_checkpoint_model, predict_image

st.set_page_config(page_title="AI403 Handwritten Digit Recognition", page_icon="🔢", layout="centered")
st.title("🔢 AI403 — Handwritten Digit-String Recognition")
st.caption("M5 Live Demo | ORAND-CAR-2014 | CNN / sequence models + CTC")
st.info("Upload a cropped cheque courtesy-amount image. The output is a digit string; leading zeros are preserved as text.")

model_label = st.selectbox("Choose model architecture", list(MODEL_CHECKPOINTS))
uploaded = st.file_uploader("Upload handwritten amount image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

model_name, _ = MODEL_CHECKPOINTS[model_label]
seed = st.selectbox("Checkpoint seed", [7, 21, 42], index=0)
checkpoint_path = checkpoint_for(model_label, seed)

if model_label == "Baseline CNN":
    st.warning("Baseline is included as a pipeline demonstration. Its recorded validation CER is about 78.36% and exact-match accuracy is 0%; do not treat its output as reliable.")
else:
    st.success("A trained checkpoint for this architecture is bundled with this project.")

if uploaded is not None:
    image = Image.open(uploaded).copy()
    st.image(image, caption="Uploaded input", use_container_width=True)
    if st.button("Recognize digit string", type="primary"):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with st.spinner(f"Loading {model_label} checkpoint and running inference..."):
                model, checkpoint, device = load_checkpoint_model(model_name, checkpoint_path, device)
                prediction, processed, width = predict_image(image, model, device)
            st.subheader("Recognition result")
            if prediction:
                st.metric("Predicted digit string", prediction)
            else:
                st.warning("The model returned an empty digit string. Try a clearer, tightly cropped image.")
            st.caption(f"Checkpoint seed: {checkpoint['config'].get('seed', 'unknown')} · Device: {device} · Resized width: {width}px")
            if model_name == "baseline":
                st.warning("This is a functional end-to-end prediction, but the Baseline's validation performance is very weak.")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
else:
    st.caption("Choose a model and upload one cropped image to enable recognition.")

with st.expander("Model and evaluation notes"):
    st.markdown("""
    - The application loads the selected model's trained checkpoint; it does not train models during demo use.
    - Images are converted to grayscale, resized to height 48 while preserving aspect ratio, and normalized to [0, 1].
    - The CTC decoder collapses repeated adjacent frame predictions and removes blank tokens.
    - Leading zeros are preserved because the output is a string, not an integer.
    - Validation metrics are reported separately from the M4-supplied official-test evidence.
    """)

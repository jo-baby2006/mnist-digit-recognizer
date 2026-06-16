import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="centered"
)

st.title("✏️ Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) in the box below and the AI will predict it!")

@st.cache_resource
def load_model():
    model = keras.models.load_model("best_mnist_model.keras")
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading failed: {e}")

st.subheader("Draw your digit here:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if predict_btn and canvas_result.image_data is not None and model_loaded:
    img = canvas_result.image_data

    img_pil = Image.fromarray(img.astype("uint8"), mode="RGBA").convert("L")
    img_pil = img_pil.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img_pil) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    predictions = model.predict(img_array)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    st.divider()
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Predicted Digit", value=str(predicted_digit))
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

    with res_col2:
        st.markdown("**Confidence for all digits:**")
        for i, prob in enumerate(predictions):
            bar_label = f"{i}  {'← predicted' if i == predicted_digit else ''}"
            st.progress(float(prob), text=f"{bar_label}  {prob*100:.1f}%")

    if confidence < 60:
        st.warning("⚠️ Low confidence — try drawing the digit more clearly.")

with st.sidebar:
    st.header("✏️ Tips for best results")
    st.markdown("""
- Draw the digit **large** and **centered**
- Use **thick strokes**
- Keep it simple — one digit at a time
- If prediction is wrong, try drawing more clearly
    """)
    st.divider()
    st.caption("Built with Keras + Streamlit")
    st.caption("Model trained on MNIST dataset (99%+ accuracy)")

import os
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras import layers
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="centered"
)

st.title("✏️ Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) in the box below and the AI will predict it!")

# ── Rebuild the exact same model architecture ──────────────
@st.cache_resource
def load_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.load_weights("mnist_weights.weights.h5")
    return model

try:
    model = load_model()
    model_loaded = True
    st.success("✅ Model loaded successfully!")
except Exception as e:
    model_loaded = False
    st.error(f"Model loading failed: {e}")

# ── Drawing Canvas ─────────────────────────────────────────
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

# ── Prediction ─────────────────────────────────────────────
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

# ── Sidebar ────────────────────────────────────────────────
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

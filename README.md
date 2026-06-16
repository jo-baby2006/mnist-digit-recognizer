Handwritten Digit Recognizer

A deep learning web app that recognizes handwritten digits (0–9) in real time.  
Built with **TensorFlow/Keras** for the model and **Streamlit** for the web interface.

---

 Live Demo

 **[Try it here](https://jo-baby2006-mnist-digit-recognizer-app.streamlit.app)**

Draw any digit in the canvas → click Predict → see the result instantly!

---

How It Works

```
User draws a digit (0–9)
        ↓
Canvas captures 280×280 image
        ↓
App resizes to 28×28 pixels
        ↓
Normalizes pixel values (0–255 → 0.0–1.0)
        ↓
Feeds into trained CNN model
        ↓
Model outputs confidence for each digit
        ↓
Predicted digit shown with confidence %
```

---

 Model Architecture

A custom Convolutional Neural Network (CNN) trained on the MNIST dataset.

| Layer | Details |
|---|---|
| Conv2D Block 1 | 32 filters, 3×3, BatchNorm, ReLU |
| Conv2D Block 2 | 64 filters, 3×3, BatchNorm, ReLU |
| Conv2D Block 3 | 128 filters, 3×3, BatchNorm, ReLU |
| Dense Layer 1 | 256 neurons, ReLU, Dropout 0.4 |
| Dense Layer 2 | 128 neurons, ReLU, Dropout 0.3 |
| Output Layer | 10 neurons, Softmax |

 Training Details
- **Dataset:** MNIST (60,000 training / 10,000 test images)
- **Optimizer:** Adam (lr=0.001)
- **Data Augmentation:** rotation, zoom, shifts
- **Callbacks:** EarlyStopping, ReduceLROnPlateau
- **Test Accuracy:** ~99%+

---

 Project Structure

```
mnist-digit-recognizer/
├── app.py                      ← Streamlit web app
├── requirements.txt            ← Python dependencies
├── mnist_weights.weights.h5    ← Trained model weights
└── README.md                   ← This file
```

---

 Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/jo-baby2006/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```

**Step 4 — Open in browser**
```
http://localhost:8501
```

---

 Dependencies

| Library | Purpose |
|---|---|
| streamlit | Web app framework |
| keras | Neural network |
| jax | Backend for Keras |
| numpy | Array operations |
| Pillow | Image processing |
| streamlit-drawable-canvas | Drawing canvas widget |

---

Tips for Best Results

- Draw the digit **large** and **centered** in the box
- Use **thick strokes**
- Keep it simple — one digit at a time
- If prediction is wrong, try drawing more clearly

---

 What I Learned

- Image preprocessing (normalization, reshaping)
- CNN architecture (Conv2D, MaxPooling, BatchNorm, Dropout)
- Data augmentation to improve accuracy
- Model training with callbacks (EarlyStopping, ReduceLROnPlateau)
- Deploying a deep learning model with Streamlit
- Hosting on Streamlit Community Cloud for free

---


## 📄 License

MIT License — free to use and modify.

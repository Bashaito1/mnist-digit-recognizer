import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
from PIL import Image
import psycopg2
from datetime import datetime
import numpy as np
import pandas as pd

DB_URL = "postgresql://neondb_owner:npg_d1om0Aybvchj@ep-quiet-pine-abg01685-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🖌️ MNIST Digit Recognizer with PostgreSQL Logging")
st.markdown("Draw a digit (0-9) below, then click **Predict**.")


# ------------------------------------------------
# 1. Logging function: connect to PostgreSQL and insert a row
# ------------------------------------------------

def log_prediction(predicted_digit: int, true_label: int):
    DB_URL = "postgresql://neondb_owner:npg_d1om0Aybvchj@ep-quiet-pine-abg01685-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"



    try:
        st.info("Connecting to the database...")
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        st.info(f"Logging: predicted={predicted_digit}, true={true_label}")
        cur.execute(
            "INSERT INTO predictions (predicted_digit, true_label, timestamp) VALUES (%s, %s, %s);",
            (predicted_digit, true_label, datetime.utcnow())
        )

        conn.commit()
        cur.close()
        conn.close()
        st.success("✅ Prediction logged successfully.")
    except Exception as e:
        st.error("❌ Failed to log prediction.")
        st.exception(e)


def show_recent_predictions():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        cur.execute("SELECT predicted_digit, true_label, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if rows:
            st.subheader("🧠 Recent Predictions")
            df = pd.DataFrame(rows, columns=["Predicted", "True Label", "Timestamp"])
            st.dataframe(df)
        else:
            st.info("No predictions logged yet.")

    except Exception as e:
        st.error(f"❌ Failed to retrieve predictions: {e}")


# ------------------------------------------------
# 2. Define the model
# ------------------------------------------------
# Define the SimpleNN model architecture matching your saved model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x=x.view(-1,28*28)
        return self.fc(x)

@st.cache_resource
def load_model():
    model = SimpleNN()
    model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

# ------------------------------------------------
# 3. Preprocessing: convert PIL image → model tensor
# ------------------------------------------------
    
def preprocess_image(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(img)
    tensor = tensor.view(-1, 28 * 28)
    return tensor

# ------------------------------------------------
# 4. UI: Canvas + Prediction + Logging
# ------------------------------------------------

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert("RGB")
    st.image(img, caption="Your Drawing", width=140)

    if st.button("Predict"):
        gray_img = img.convert("L")
        input_tensor = preprocess_image(gray_img)
        model = load_model()

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            predicted_digit = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_digit].item()

        st.session_state.predicted_digit = predicted_digit  # Save to session state
        st.session_state.confidence = confidence

        st.success(f"Predicted Digit: {predicted_digit} ({confidence:.2%} confidence)")

        # Only show label input if prediction exists
if "predicted_digit" in st.session_state:
    true_label = st.number_input(
        "Enter the correct digit (0-9):",
        min_value=0, max_value=9, step=1,
        key="true_label_input"
    )

    if st.button("Submit True Label"):
        log_prediction(st.session_state.predicted_digit, true_label)
        st.divider()
        show_recent_predictions()


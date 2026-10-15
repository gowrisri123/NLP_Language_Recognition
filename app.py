import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="NLP Language Recognition",
    page_icon="🌐",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("language_model.joblib")

st.title("🌐 NLP Language Recognition")
st.write(
    "Enter a sentence to identify its language "
    "using our trained machine learning model."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run python train_model.py first.")
    st.stop()

accuracy_file = Path("model_accuracy.txt")

if accuracy_file.exists():
    try:
        accuracy = float(
            accuracy_file.read_text(encoding="utf-8").strip()
        )
        st.metric("Test-set accuracy", f"{accuracy:.2f}%")
        st.caption(
            "Measured on a small held-out test set; "
            "real-world accuracy may differ."
        )
    except ValueError:
        st.info("Run train_model.py to refresh the evaluation.")
else:
    st.info("Run train_model.py to calculate test accuracy.")

text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: నమస్కారం మీరు ఎలా ఉన్నారు",
    height=120
)

if st.button("Detect Language", type="primary"):
    if len(text.strip()) < 3:
        st.warning("Please enter a longer sentence.")
    else:
        prediction = model.predict([text.strip()])[0]

        st.success("Language detected!")
        st.subheader("Recognition Result")
        st.write(f"**Detected Language:** {prediction}")

st.divider()
st.caption(
    "Languages: English, Telugu, Hindi, French, "
    "Spanish, Tamil and Kannada."
)
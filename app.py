from segment import segment_digits
import streamlit as st
import tempfile
import os
import cv2

from PIL import Image
from recognise import recognize

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="🔢",
    layout="wide"
)

# --------------------------------
# HEADER
# --------------------------------

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "An AI-powered handwritten numerical recognition "
    "system using PyTorch, OpenCV and Feature Fusion."
)

st.divider()

# --------------------------------
# MODEL INFORMATION
# --------------------------------

st.sidebar.title("Model Information")

st.sidebar.success("Model loaded")

st.sidebar.write("Architecture: CNN + MLP")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Dataset: MNIST")
st.sidebar.write("Inference: GPU / CPU")

# --------------------------------
# IMAGE UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload handwritten digits",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------
    # RECOGNITION
    # --------------------------------

    if st.button(
        "Recognize Digits",
        type="primary",
        use_container_width=True
    ):

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp:

            image.save(temp.name)
            temp_path = temp.name

        try:

            with st.spinner("Analyzing handwriting..."):

                number, predictions, output = recognize(
                    temp_path
                )

            # Display detected image
            output_rgb = cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            )

            with col2:

                st.subheader("Individual Segmented Digits")

                digits, _ = segment_digits(temp_path)

                cols = st.columns(5)

                for i, digit in enumerate(digits):
                    with cols[i % 5]:
                        st.image(
                            digit,
                            caption=f"Region {i + 1}",
                            width=80
                        )

            st.divider()

            # Final prediction
            st.subheader("Recognition Result")

            st.success(
                f"Recognized Number: {number}"
            )

            # Individual predictions
            st.subheader("Digit-wise Predictions")

            columns = st.columns(
                max(1, len(predictions))
            )

            for i, item in enumerate(predictions):

                with columns[i]:

                    st.metric(
                        label=f"Digit {i+1}",
                        value=item["digit"],
                        delta=f"{item['confidence']*100:.2f}% confidence"
                    )

        except Exception as e:

            st.error(f"Recognition failed: {e}")

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

else:

    st.info(
        "Upload an image containing handwritten numbers to begin."
    )

st.divider()

st.caption(
    "Powered by PyTorch | OpenCV | MNIST | Feature Fusion Network"
)

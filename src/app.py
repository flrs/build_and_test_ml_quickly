"""Streamlit app for trash recognition.

In contrast to the simple version (`app_simple.py`), this app includes:
- Additional information about the data and model
- Feedback functionality
- Caching which improves the user experience since less content needs to be reloaded
"""

import streamlit as st
from PIL import Image
from streamlit_feedback import streamlit_feedback

from ml.model import TrashRecognizerModel
from utils.feedback import log_feedback

# Configure streamlit
st.set_page_config(
    page_title="Trash Recognizer",
    page_icon="🗑️",
    menu_items={
        "Get help": None,
        "Report a Bug": "mailto:Email for bug reports" if "@" in "Email for bug reports" else None,
        "About": "Demo app for PyData Conference Berlin 2024",
    },
)


@st.cache_resource
def load_model():
    # Call model
    model = TrashRecognizerModel()
    return model


model = load_model()

# App content
st.markdown("# Trash Recognizer")
st.markdown(
    """Detect trash in images and predict into which trash can it should be disposed.
    """
)
with st.expander("ℹ️ About the data and model"):
    st.markdown(
        """
        The model was trained on the [TACO dataset](http://tacodataset.org/), see: 
        > Proença, P. F., & Simões, P. (2020).
         TACO: Trash Annotations in Context for Litter Detection. *arXiv Preprint arXiv:2003.06975*. 
         
         The model is a Mask R-CNN model with a ResNet-50-FPN backbone, trained through Azure Machine Learning. 
        """
    )

# Add a file uploader
uploaded_files = st.file_uploader(
    "Choose a file", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Check if any files have been uploaded
if len(uploaded_files) > 0:
    # Check if there is a new set of files – if so, process them
    if not set(st.session_state.get("results", {}).keys()) == set(
        [file.name for file in uploaded_files]
    ):
        with st.spinner(f"Processing {len(uploaded_files)} image(s)..."):
            # Convert each uploaded file to a numpy array
            # (this is needed because model.predict expects a list of numpy arrays)
            images = [Image.open(image).convert('RGB') for image in uploaded_files]
            # Predict the type of trash in each image
            predictions = model.predict(images)
            # Store the results in the session state
            st.session_state["results"] = {
                # Each result is a tuple of the image and its prediction
                file.name: [image, prediction]
                for image, prediction, file in zip(images, predictions, uploaded_files)
            }
        # Rerun the app so that subsequent user-triggered state changes in the frontend
        # app, like submitting feedback, do not trigger an app rerun.
        st.rerun()
else:
    # If no files have been uploaded, clear the results in the session state
    st.session_state["results"] = {}

# Display classification results
for image_name, result in st.session_state.get("results", {}).items():
    image, prediction = result
    col1, col2 = st.columns(2)

    with col1:
        plot_fig = image
        if len(prediction[0]) > 0:
            # Trash detected
            plot_fig = model.plot_predictions(image, prediction)
        st.image(plot_fig, use_column_width=True)

    with col2:
        if len(prediction[0]) == 0:
            st.write("**🪴 No trash detected.**")
        else:
            st.write("🗑️ **Detected Trash**")
            for predicted_class in set(prediction[1]):
                st.write(
                    f" - {prediction[1].count(predicted_class)} item(s) for the {predicted_class} trash can"
                )

        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="Any feedback?",
            on_submit=log_feedback,
            args=("recognition", image_name),
            align="center",
            key=f"feedback_recognition_{image_name.replace('.','')}",
        )

"""Simple version of the app shown in the PyData talk."""
import streamlit as st
from PIL import Image

from ml.model import TrashRecognizerModel

st.title('Trash Recognizer')

model = TrashRecognizerModel()
uploaded_files = st.file_uploader('Upload image(s)', accept_multiple_files=True)
if uploaded_files:
    images = [Image.open(image).convert('RGB') for image in uploaded_files]
    results = model.predict(images)

    for image, result in zip(images, results):
        col1, col2 = st.columns(2)

        if len(result[0]) == 0:
            # No Trash detected
            with col1:
                st.image(image)
            with col2:
                st.write('**🪴 No trash detected.**')
        else:
            # Trash detected
            with col1:
                st.image(model.plot_predictions(image, result))
            with col2:
                st.write('🗑️ **Detected Trash**')
                for predicted_class in set(result[1]):
                    st.write(
                        f' - {result[1].count(predicted_class)} item(s) for '
                        f'the {predicted_class} trash can'
                    )
import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

import tensorflow as tf
import cv2
import face_recognition

import warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(
    page_title = 'DataFlow App',
    page_icon = '📊',
    layout = 'wide',
    initial_sidebar_state = 'auto',
)

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('models/model.h5')
    return model

def preprocess_image(image):
    img = cv2.resize(image, (224, 224)) 
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# Home Page Content
def main():
    model = load_model()
    threshold = 0.33

    with st.sidebar:
        # Set Sidebar Content
        st.sidebar.image('media/logo.png', use_column_width=True)
        with st.container():
            img_uploaded = st.file_uploader("Choose an image...", type=["jpg", "png"])
        st.info('ReDeepFake is an advanced Deepfake detection model for 2D flat images.')
        st.caption('Made with ❤️ by [DataFlow](https://dataflow.kz) team.')
    
    if img_uploaded is not None:
        with st.spinner('Processing the image, getting faces...'):
            image = cv2.imdecode(np.frombuffer(img_uploaded.read(), dtype=np.uint8), 1)
            face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            st.warning('Faces not found!')
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            with st.spinner(f'Preprocessing the face #{i+1}:'):
                processed_face = preprocess_image(face_image)
                processed_face = np.expand_dims(processed_face, axis=0)

                prediction = model.predict(processed_face)

                predicted_class = "FAKE" if prediction[0, 0] > threshold else "REAL"

                st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), caption=f"Face {i+1}: {predicted_class} | Score: {prediction[0, 0]:.2f}", width=350)

                if predicted_class == "FAKE":
                    st.warning('The image is most likely fake!')
                else:
                    st.success('The image is most likely real!')
                

if __name__ == "__main__":
    main()


import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv

import tensorflow as tf
import cv2
import face_recognition

import warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(
    page_title='ReDeepFake Demo App',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='auto',
)


@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('.models/model.h5')
    return model


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# Home Page Content
def main():
    model = load_model()

    with st.sidebar:
        # Set Sidebar Content
        st.sidebar.image('media/logo.png', use_column_width=True)
        with st.container():
            img_uploaded = st.file_uploader(
                "Choose an image...", type=["jpg", "png"])
            threshold = st.select_slider(
                'Threshold', options=[i/100 for i in range(0, 101, 5)], value=0.5)
        st.info(
            'ReDeepFake is an advanced Deepfake detection model for 2D flat images.')
        st.caption('Made with ❤️ by [DataFlow](https://dataflow.kz) team.')

    st.title('🧑 ReDeepFake v1.3')
    st.markdown('''
        This is a demo app for ReDeepFake model - Advanced Deepfake detection model for 2D flat images.
    ''')
    colA, colB = st.columns(2)
    colA.markdown('''
        **How to use this app?**
        1. Upload an image.
        2. Adjust the threshold.
        3. Image will be processed and real/fake faces will be detected automatically.            
    ''')
    colB.markdown('''
        **Resources:**
        * **Kaggle Notebook:** [ReDeepFake](https://www.kaggle.com/code/armanzhalgasbayev/deepfake-detection-efficientnetb4-tf-cnn)
        * **GitHub Repository:** [ReDeepFake](https://github.com/silvermete0r/redeepfake-demo-app)
        * **Hugging Face Demo App:** [ReDeepFake](https://huggingface.co/spaces/dataflow/redeepfake-demo)
        * **Download Model:** [ReDeepFake](https://huggingface.co/dataflow/redeepfake)
    ''')
    st.caption('**Note:** Before using the model read about limitations.')
    with st.expander('Limitations', expanded=False):
        st.markdown('''
            1. The model's performance may be influenced by variations in lighting conditions, image quality, and diverse facial expressions.

            2. It may not be fully robust against emerging deepfake generation techniques: Modern image generation methods use advanced descriptors to assess the quality of the image's realism, so photos of such faces are difficult to distinguish from real people. 

            3. Deepfakes made by using 3D image processing technologies and manually modified images by the authors cannot be recognized correctly by the model.
        ''')

    if img_uploaded is not None:
        with st.spinner('Processing the image, getting faces...'):
            image = cv2.imdecode(np.frombuffer(
                img_uploaded.read(), dtype=np.uint8), 1)
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

                predicted_class = "FAKE" if prediction[0,
                                                       0] > threshold else "REAL"

                st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB),
                         caption=f"Face {i+1}: {predicted_class} | Score: {prediction[0, 0]:.2f}", width=350)

                download_img = cv2.imencode('.png', face_image)[1].tobytes()
                st.download_button(label="Download Image", data=download_img,
                                   file_name=f"face_{i+1}_{predicted_class}.png", mime="image/png")

                if predicted_class == "FAKE":
                    st.warning('The image is most likely fake!')
                else:
                    st.success('The image is most likely real!')

                st.divider()


if __name__ == "__main__":
    main()
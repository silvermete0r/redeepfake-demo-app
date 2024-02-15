![DataFlow_v1 0](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/3f6d3b6e-fa65-4cc1-b703-383bbe063599)

# ReDeepFake Model Card

![Static Badge](https://img.shields.io/badge/tensorflow-black?style=flat&logo=tensorflow) 
[![GitHub stars](https://img.shields.io/github/stars/silvermete0r/redeepfake-demo-app.svg?style=flat&logo=github&colorB=yellow)](https://github.com/silvermete0r/redeepfake-demo-app/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/silvermete0r/redeepfake-demo-app.svg?style=flat&logo=github&colorB=blue)](https://github.com/silvermete0r/redeepfake-demo-app/network/)
[![GitHub issues](https://img.shields.io/github/issues/silvermete0r/redeepfake-demo-app.svg?style=flat&logo=github)](https://github.com/silvermete0r/redeepfake-demo-app/issues)
[![GitHub license](https://img.shields.io/github/license/silvermete0r/redeepfake-demo-app.svg?style=flat&logo=github&colorB=green)](https://github.com/silvermete0r/redeepfake-demo-app/blob/master/LICENSE)
![Static Badge](https://img.shields.io/badge/Powered%20by-Dataflow-lime?style=plastic)

*Before using this app download the model via link [HuggingFace](https://huggingface.co/dataflow/redeepfake)*

## Overview

**Model Name:** `ReDeepFake` 

**Task:** Advanced Deepfake detection model for 2D flat images.

**Model Type:** Convolutional Neural Network (CNN) -> [EfficientNetB4](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet) Model Architecture based.

**Framework:** `TensorFlow`

**Current Version:** V1.3

**Key Study:** [Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/competitions/deepfake-detection-challenge) (in 2019-2020, by Meta, AWS, Microsoft and AI’s Media Integrity Steering Committee).

**Key Parthner:** `Microsoft Imagine Cup 2024`

**Kaggle Notebook:** https://www.kaggle.com/code/armanzhalgasbayev/deepfake-detection-efficientnetb4-tf-cnn

**Research Paper:** https://www.researchgate.net/publication/341903582_Deepfake_Video_Detection_Using_Convolutional_Neural_Network

**ReDeepFake for video Processing:** [video_process.py](https://github.com/silvermete0r/redeepfake-demo-app/blob/master/video_process.py)

## Description

ReDeepFake is a deep learning model designed for the purpose of detecting deepfake content within 2D flat images. Leveraging EfficientNetB4 advanced CNN architecture, the model is trained to discern subtle patterns indicative of deepfake manipulation.

## Key Features

- **Architecture:** Utilizes EfficientNetB4 model architecture with a custom-designed additional layers. ReDeepFake model architecture shown below in the figure:

![ReDeepFake - ReDeepFake Model Architecture](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/df7e1ba4-2da7-4b78-bde2-c031ba52e173)

- **Training Data:** Pre-processed dataset of 224x224 resized real/fake image from DFDC competition was used for training our `ReDeepFake` model.
  - Dataset of 32,000 face images:
    - Real: 16,000;
    - Fake: 16,000;
![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/f50b2107-78fd-4458-9c4d-8f23fc9056d0)

- **Performance Metrics:** Evaluated based on `Accuracy` and `F1-score`.
  - **Accuracy:** The proportion of correctly classified images.
![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/79798e97-2261-421d-a512-e888c2401fec)

  - **F1-score:** The harmonic mean of precision and recall.
    - **Precision:** The ratio of true positive predictions to the total predicted positives.
    - **Recall:** The ratio of true positive predictions to the total actual positives.
![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/327d5060-fb80-4c66-a3c5-f9b94e601386)


- **Dataset Source:** [deepfake_faces](https://www.kaggle.com/datasets/dagnelies/deepfake-faces)

- **Deepfake Detection Challenge:** [Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge)

- **Final Trained Model Link:** [ReDeepFake v1.3](https://github.com/silvermete0r/redeepfake-demo-app/blob/master/.models/model.h5)

- **Generation of Deepfake images based on ONNX & PyTorch:** https://gist.github.com/silvermete0r/e24f35df5b9a62f03a7e73d1f3d448c3

- **Try using our demo:** https://huggingface.co/spaces/dataflow/redeepfake-demo

![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/3dd74781-9d0b-42fd-ab6e-eb3ce2063769)

## Model Performance

![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/adefa62a-fc3a-48d2-a141-9d04e8607240)

* **Test Loss:** `0.4872`

* **Test Accuracy:** `0.9022`

![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/69a305bb-b5d6-42e4-be94-39cecb9d05c3)

![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/db22e81d-d0f0-4722-a40f-d763aaeb8ff3)

* **Test on random set of 20 images:** 

![image](https://github.com/silvermete0r/redeepfake-demo-app/assets/108217670/bfc7112e-2a60-4e8d-bd9b-d3a39bea837c)


## Usage

### Prerequisites

* face_recognition<=1.3.0
* numpy<=1.23.5
* opencv_python<=4.7.0.72
* pandas<=2.0.2
* tensorflow==2.15.0.post1
* streamlit<=1.23.1 (Optional - for demo)

### Areas of Usage

1. **Media Integrity Verification:**
   - Verify the authenticity of images and videos in the media industry to ensure integrity and trustworthiness.

2. **Social Media Content Moderation:**
   - Enhance content moderation on social media platforms by detecting and preventing the spread of deepfake content.

3. **Security and Surveillance:**
   - Strengthen security and surveillance systems by identifying manipulated visuals that could compromise the accuracy of facial recognition and surveillance technologies.
   

### Loading the Model

```python
import tensorflow as tf

## Load the ReDeepFake model
redeepfake_model = tf.keras.models.load_model('redeepfake_model.h5')
```

### Preprocessing the Input Image
```python
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img
```

### Making Predictions

```python
# 'image' -> preprocessed input image data
prediction = redeepfake_model.predict(image)
```

## Model Limitations

1. The model's performance may be influenced by variations in lighting conditions, image quality, and diverse facial expressions.

2. It may not be fully robust against emerging deepfake generation techniques: Modern image generation methods use advanced descriptors to assess the quality of the image's realism, so photos of such faces are difficult to distinguish from real people. 

3. Deepfakes made by using 3D image processing technologies and manually modified images by the authors cannot be recognized correctly by the model.

## Ethics

1. **Informed Consent:**
   - Ensure that individuals involved in the data used for training and testing the model have given informed consent for their images to be used. Respect privacy and obtain explicit permission when necessary.

2. **Responsible Disclosure:**
   - If the model identifies potential deepfake content related to an individual, exercise responsible disclosure. Notify the concerned parties or relevant authorities in a timely and responsible manner, allowing them to address the situation appropriately.

3. **Avoid Unlawful Use:**
   - Do not use the ReDeepFake model for any activities that violate laws, regulations, or ethical standards. Prohibit the use of the model for malicious purposes, such as generating harmful deepfake content or engaging in activities that infringe on privacy rights.

**License**

This model is provided under the [LICENSE](LICENSE) file. Please review the terms and conditions before usage.

**Issues and Contributions**

For issues or contributions, please visit the [GitHub repository](https://github.com/silvermete0r/redeepfake-demo-app).

**Acknowledgments**

Special thanks to the contributors and the open-source community for their support and collaboration.


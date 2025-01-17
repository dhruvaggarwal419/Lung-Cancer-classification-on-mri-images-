import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import streamlit as st 
import numpy as np
from keras.models import load_model
# Ensure eager execution is enabled
tf.compat.v1.enable_eager_execution()
st.header('lung cancer Classification using CNN Model')
lungcancer_names = ['cancer', 'no_canccer']

# Load the model
model = load_model('lung_cancer_classification_aarchi.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + lungcancer_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

#upload and classify 
uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, width=200)
    
    # Get the path of the saved file
    saved_image_path = os.path.join('upload', uploaded_file.name)
    
    # Display the classification result
    st.markdown(classify_images(saved_image_path))
import numpy as np
import pandas as pd
import streamlit as st
import base64
# import tensorflow as tf
import keras
from PIL import Image
from util import classify 



st.title("Pneumonia Detector")
st.header("Upload an image of chest X-ray")

file=st.file_uploader("Upload", type=["jpeg", "jpg", "png"])


model = keras.models.load_model("./pneumonia_classifier.h5")
with open("./labels.txt", "r") as f:
    classnames = [a[:-1].split(" ")[1] for a in f.readlines()]
    f.close()
# print(classnames)
if file is not None:
    image=Image.open(file).convert("RGB")
    st.image(image,use_column_width=True)
    
    classname,conf_score=classify(image,model,classnames)

    st.write("## {}".format(classname))
    st.write("### Score: {}".format(conf_score))
    
    
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./background.png') 
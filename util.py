import streamlit as st
import base64
import numpy as np
from PIL import ImageOps,Image




def classify(image,model,classnames):
    
    image=ImageOps.fit(image,(224,224),Image.Resampling.LANCZOS)
    image_array=np.asanyarray(image)
    normalizeimagearray=image_array.astype(np.float32)/127.5 -1
    data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
    data[0]=normalizeimagearray
    
    
    # predict
    
    predict=model.predict(data)
    # index=np.argmax(predict)
    index = 0 if predict[0][0]>0.95 else 1
    class_name=classnames[index]
    confi_score=predict[0][index]
    
        
    
    return class_name,confi_score
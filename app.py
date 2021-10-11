import streamlit as st
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import tensorflow 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 
fig=plt.figure()
st.title('Surface Crack Detection')


def predict(image):
  c_model='ocrack.h5'
  IMAGE_SHAPE=(200,200,3)
  models=load_model(c_model,compile=False,custom_objects={'KerasLayer': hub.KerasLayer})
  test_image=image.resize((200,200))
  test_image=img_to_array(test_image)
  test_image=test_image/255.0
  test_image=np.expand_dims(test_image,axis=0)
  classes=['Negative','Positive']
  predictions=models.predict(test_image)
  scores=tf.nn.softmax(predictions[0])
  results={
      'Negative':0,
      'Positive':1
  }
  results=f"{classes[np.argmax(scores)]} with a {(100*np.max(scores)).round(2)}% confidence"
  return results


def main():
  upload=st.file_uploader('UPLOAD IMAGE',type=['jpg'])
  btn=st.button('Predict')
  if upload is not None:
    image=Image.open(upload)
    st.image(image,caption='Uploaded')
  if btn:
    if upload is None:
      st.write('Upload Image!!!')
    else:
      with st.spinner('Model Loading...'):
        plt.imshow(image)
        plt.axis('off')
        predictions=predict(image)
        st.success('Predicted')
        st.write(predictions)
        st.pyplot(fig)

if __name__=="__main__":

  main()

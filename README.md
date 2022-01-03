
# Crack Detection App

 The Crack Detection App is built using a Deep Learning Model. This app is
 used to detect cracks on any surface .
*******************************************************************
## Architecture
 To develop the model first we have augmented the images so that our 
 model can learn on variations of the images that can improve the ability of the fit models to generalize what they have learned to new images.

 
 We have used custom CNN Architecture from scratch to develop the deep learning
 model and then with this CNN Architecture using Adam Optimizer we 
 have trained our model. After our model got trained we got an training 
 accuracy of 90.1% and testing accuracy of 85.55% respectively.

## Web App Development
 Our model got developed and then we tested the model using some
 test images. We used Gradio to do a demo testing of DL Model and finally
 we used Streamlit to build the Deep Learning Web App. Then we 
 deployed this Streamlit app on Heroku.

*******************************************************************





## Run the app on local machine

Download the app.py and run the app.py

```bash
  streamlit app.py
```
*******************************************************************
## 🔗 Deployment App
[![Streamlit](https://img.shields.io/badge/STREAMLIT-565?style=for-the-badge&logo=ko-fi&logoColor=white)](https://crack-detection-app.herokuapp.com/)
*******************************************************************
## Demo Video
![Demo Video](./Crack.gif)



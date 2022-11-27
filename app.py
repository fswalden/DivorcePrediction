import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image

def convertInput(val, positive = True):
    """
    Values converted to 0-4 scale. Negative flag corresponds to input question: 
    If answering "Often" or "Always" to that question has a negative meaning
    regarding a relationship, the scale is flipped

    0 is seen as a positive answer. Higher numbers are negative.
    """
    if val == 'Never': 
        return 4 if positive else 0
    elif val == 'Rarely':
        return 3 if positive else 1
    elif val == 'Sometimes':
        return 2
    elif val == 'Often':
        return 1 if positive else 3
    else:
        return 0 if positive else 4


def get_user_input():
    # Get multiple choice user input using streamlit's radio feature
    
    atr40 = convertInput(st.radio("We begin arguing before I understand the problem.", ["Never", "Rarely", "Sometimes", "Often", "Always"], index=2), False)
    atr17 = convertInput(st.radio("We share similar views about being happy in our life together.", ["Never", "Rarely", "Sometimes", "Often", "Always"], index=2))
    atr19 = convertInput(st.radio("We share similar views about what the roles in our marriage should be.", ["Never", "Rarely", "Sometimes", "Often", "Always"], index=2))
    atr18 = convertInput(st.radio("We share similar ideas about what a marriage should be.", ["Never", "Rarely", "Sometimes", "Often", "Always"], index=2))

    features = {
        'Atr40': atr40,
        'Atr17': atr17,
        'Atr19': atr19,
        'Atr18': atr18
    }
    
    data = pd.DataFrame(features,index=[0])

    return data

# Load the model
model = joblib.load("MLP_Model.sav")

st.title("Divorce Predictor")

#read in wine image and render with streamlit
image = Image.open('DivorcePicture.jpeg')
st.image(image, caption='',use_column_width=True)

st.header("Answer the following questions about your marriage:")

data = get_user_input()

button = st.button("Make prediction")

if button:
    # Predict divorce or marriage
    outVal = model.predict(data)[0]

    if outVal == 1:
        outVal = "The model predicts a **divorce**."
    else:
        outVal = "The model predicts a **successful marriage!**"

    st.write(outVal)
else:
    st.write("Answer the above questions.")

# Import the required packages
import numpy as np
import streamlit as st
import pickle
import pandas as pd

# Create a browser header 
st.set_page_config(page_title='Iris Project - Utkarsh')

# Add the title to the page
st.title('Iris Project - Utkarsh Gaikwad')

# Take sep_len, sep_wid ... as input from user
sep_len = st.number_input('Sepal Length : ', min_value=0.00, step=0.01)
sep_wid = st.number_input('Sepal Width : ', min_value=0.00, step=0.01)
pet_len = st.number_input('Petal Length : ', min_value=0.00, step=0.01)
pet_wid = st.number_input('Petal Width : ', min_value=0.00, step=0.01)

# Create a submit button
submit = st.button('Predict')

st.subheader('Predictions are :')

def predict_data():
    xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T 
    xnew.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # Load the pre
    with open('notebook/pipe.pkl', 'rb') as file1:
        pre = pickle.load(file1)
    xnew_pre = pre.transform(xnew)
    # Load the model
    with open('notebook/model.pkl', 'rb') as file2:
        model = pickle.load(file2)
    # Provide the predictions
    pred = model.predict(xnew_pre)
    # Predict the probability
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    return pred, max_prob

if submit:
    pred, max_prob = predict_data()
    st.subheader(f'Predicted Species is : {pred[0]}')
    st.subheader(f'Probability of prediction : {max_prob:.4f}')
    st.progress(max_prob)

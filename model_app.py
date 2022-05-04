# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:12:04 2022

@author: navan
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf

#Loading the model
regressor=tf.keras.models.load_model('stock_model.hdf5')

#loading the scalar from pickle file
pickle_in=open("sc.pkl","rb")
sc=pickle.load(pickle_in)

def predict(df):
    df=np.array(df)
    df=sc.transform(df)
    df=np.reshape(df,(1,60,1))
    res=regressor.predict(df)  
    return sc.inverse_transform(res) 
 
    

def main():
    
    st.title("Stock Price Predictor")

     

    uploaded_file = st.file_uploader("Choose a file",type="CSV")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    if st.button("Predict"):
        result=predict(df)
        st.success('the output is {}'.format(result))
if __name__=='__main__':
    main()
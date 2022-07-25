# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:42:57 2022

@author: eshan
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

#%% Constants
MODEL_PATH = os.path.join(os.getcwd(),'model','model.pkl')

#%% Load Model
classifier = pickle.load(open(MODEL_PATH,'rb'))

#%% Model Evaluation Using Test Case
# Features used in model: 'cp', 'thall', 'age', 'thalachh', 'oldpeak'
new_data=[3,1,65,158,2.3]

new_data = np.expand_dims(new_data, axis=0)

print(classifier.predict(new_data)) # Output predicted as 1 (correct)

#%% 

NEW_DATA_PATH = os.path.join(os.getcwd(),'dataset','test_case.txt')

new_df = pd.read_csv(NEW_DATA_PATH, sep=' ')

X_new = new_df.loc[:,['cp', 'thall', 'age', 'thalachh', 'oldpeak']]
y_new = new_df['output']

y_pred = classifier.predict(X_new)

print(classification_report(y_new,y_pred)) # 90% accuracy

#%% Streamlit App

st.header('Cardiovascular Risk Prediction App')
st.write('This app predicts the diagnosis of heart disease via a machine learning model trained using the dataset from https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset')
with st.form("cvd app"):
    age = st.slider('Age', 20, 80, 50, step=1)
    
    cp_dict = {'typical angina (0)': 0,'atypical angina (1)': 1,
               'non-anginal pain (2)': 2,'asymptomatic (3)': 3}
    cp_key = st.radio("Chest Pain type", ('asymptomatic (3)',
                                          'typical angina (0)',
                                          'atypical angina (1)',
                                          'non-anginal pain (2)'))
    cp = cp_dict[cp_key]
    
    thall_dict = {'fixed defect (1)':1,'normal / no thalassemia (2)':2,
                  'reversable defect (3)':3}
    thall_key = st.radio("Thalassemia", ('normal / no thalassemia (2)',
                                         'fixed defect (1)',
                                         'reversable defect (3)'))
    thall = thall_dict[thall_key]
    
    thalachh = st.slider('Maximum heart rate achieved', 65, 250, 100, step=1)
    
    oldpeak = st.number_input('ST depression induced by exercise relative to rest')
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        new_data = np.expand_dims([cp,thall,age,thalachh,oldpeak],axis=0) 
        outcome = classifier.predict(new_data)[0]
        if outcome == 1:
            st.subheader('Higher probability of heart disease :heavy_exclamation_mark:')
            
        else:
            st.subheader('Lower probability of heart disease :heavy_check_mark:')


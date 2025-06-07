#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


model = pickle.load(open('Dep.pkl', 'rb'))


# In[10]:


st.title('Sentiment Analysis App')


# In[4]:


user_input = st.text_input("Enter your review:")


# In[7]:


# input_data = np.array([user_input])


# In[8]:


# # Prediction
# prediction = model.predict(input_data)[0]
# probability = model.predict_proba(input_data)[0][1]


# In[9]:


# # Display output
# st.subheader("Prediction")
# st.write(f"Sentiment in this Sentence is:   {prediction}")
# # st.write(f"Survival Probability: {probability:.2f}")


# In[12]:


# Only show output after user types something
if user_input:
    input_data = np.array([user_input])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display output
    st.markdown("---")
    st.subheader("🎯 **Prediction Result**")
    st.write(f"Sentiment in this Sentence is:   {prediction}")
    st.markdown("---")
else:
    st.info("👈 Enter a review above to get the sentiment prediction.")


# In[ ]:





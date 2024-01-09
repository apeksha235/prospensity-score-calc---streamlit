import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


st.title("Prospensity Score calculator")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)


X=df.drop('SGLT2',axis=1)
y=df['SGLT2']
model=LogisticRegression()
result=model.fit(X,y)
df['Prospensity Score']= model.predict_proba(X)[:, 1]

st.write(df)
df.to_csv('new_data.csv',index=False)


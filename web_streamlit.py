import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Propensity Score Calculator")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    X = df.drop('SGLT2', axis=1)
    y = df['SGLT2']


    model = LogisticRegression()
    model.fit(X, y)


    df['Propensity Score'] = model.predict_proba(X)[:, 1]

    st.write(df)
    st.download_button(label="Download data as CSV", 
                       data=df.to_csv(index=False),
                       file_name='new_data.csv',
                       mime='text/csv')
else:
    st.write("Please upload a CSV file to proceed.")

import streamlit as st
from PIL import Image
import pandas as pd
import streamlit_pandas_profiling
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_pandas_profiling

from pydantic_settings import BaseSettings


## Pandas profiling
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

## ML Requirements
import pycaret
from pycaret.classification import setup,compare_models,pull,save_model,tune_model,plot_model


with st.sidebar:
    image=Image.open(r"D:\Dataset\AutoML\robot-7770312_1280.jpg")
    st.image(image)
    st.title('Auto Machine Learning App')
    choice=st.radio("Navigation",['Upload','Profiling',"ML","Download"])
    st.info('This application build on streamlit FrameWork. It helps you analyaze and check machine learning algorthms performance on dataset in CSV format using pandas profiling')

if os.path.exists("SourceData.csv"):
    df=pd.read_csv("SourceData.csv",index_col=None)

if choice=="Upload":
    st.title("Upload your Dataset for ML Modelling")
    file=st.file_uploader('Please upload dataset in CSV Format')
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("SourceData.csv",index=None)
        st.dataframe(df)

if choice=="Profiling":
    st.title('Automated Exploratory Data Analysis')
    profile = ProfileReport(df, title="Report")
    st_profile_report(profile)

if choice=='ML':
    st.title("Machine learning Classification Algorithm Report")
    #Data Setup
    target=st.selectbox("Select target Variable",df.columns)
    if st.button('Train Model'):
        setup(df,target=target)
        setup_df=pull()
        st.info("This is ML Model Expreiment")
        st.dataframe(setup_df)

        # Compare model performance
        best_model= compare_models()
        compare_df=pull()
        st.info('Model Comparison')
        st.dataframe(compare_df)
        best_model

        #Visualize Model Performance
        img = plot_model(
            best_model, plot="auc", display_format="streamlit", save=True
            )
        st.image(img)

        #Save Model
        save_model(best_model,'best_model')

if choice=='Download':
    with open('best_model.pkl','rb') as f:
        st.download_button('Download Model',f,"trained_model.pkl")
    pass

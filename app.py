import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64

def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="diabete_predictions.csv">Download CSV File</a>'
    return href

st.sidebar.image('2.jpg')

def main():
    st.markdown("<h1 style='text-align: center; color: brown;'>Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Diabetes Study in Cameroon</h2>", unsafe_allow_html=True)
    menu = ['Home','Analysis','Data Visualisation','Machine Learning']
    choice = st.sidebar.selectbox('Select Menu',menu)
    data = load_data('diabetes.csv')
    if choice == 'Home':
        left,middle, right = st.columns((2,3,2))
        with left:
            st.image('1.jpg', width=300)
        st.write('This is an app that will analyse diabetes Datas with some python tools that can optimize decisions')
        st.subheader('Diabetes Information')
        st.write('In Cameroon, the prevalence of diabetes in adults in urban areas is currently estimated at 6 – 8%, with as much as 80% of people living with diabetes who are currently undiagnosed in the population. Further, according to data from Cameroon in 2002, only about a quarter of people with known diabetes actually had adequate control of their blood glucose levels. The burden of diabetes in Cameroon is not only high but is also rising rapidly. Data in Cameroonian adults based on three cross-sectional surveys over a 10-year period (1994–2004) showed an almost 10-fold increase in diabetes prevalence.')

    elif choice == 'Analysis':
        
        st.subheader('Diabetes Dataset')
        st.write(data.head())

        if st.checkbox('Summary'):
            st.write(data.describe())
        
        if st.checkbox('Correlation'):
            fig = plt.figure(figsize=(15,15))
            st.write(sns.heatmap(data.corr(),annot=True))
            st.pyplot(fig)
    elif choice == 'Data Visualisation':
    
        if st.checkbox('Countplot'):
            fig1= plt.figure(figsize=(5,5))
            sns.countplot(x='Age',data=data)
            st.pyplot(fig1)

        if st.checkbox('Scatterplot'):
            fig2= plt.figure(figsize=(5,5))
            sns.scatterplot(x='Glucose',y='Age',data=data,hue='Outcome')
            st.pyplot(fig2)
    elif choice == 'Machine Learning':
        tab1, tab2, tab3 = st.tabs([":clipboard: Data",":bar_chart: Visualisation",":mask: Prediction"])
        upload_file= st.sidebar.file_uploader('Upload your input csv file',type=['csv'])
        if upload_file:
            df = load_data(upload_file)
            with tab1:
                st.subheader('Loaded dataset')
                st.write(df)
            with tab2: 
                st.subheader('Histogram Glucose')
                fig = plt.figure(figsize=(8,8))
                sns.histplot(x='Glucose',data=df)
                st.pyplot(fig)
            with tab3 : 
                model = pickle.load(open('model_dump.pkl','rb'))
                prediction= model.predict(df)
                st.subheader('Prediction')
                # Transformation de l'array  predit 
                pp = pd.DataFrame(prediction,columns=['Prediction'])
                ndf = pd.concat([df,pp],axis=1)
                ndf.Prediction.replace(0,'No diabetes risk',inplace= True)
                ndf.Prediction.replace(1,'Diabetes risk',inplace= True)
                st.write(ndf)
                button = st.button('Download CSV')
                if button: 
                    st.markdown(filedownload(ndf),unsafe_allow_html=True)
if __name__ == '__main__':
    main()

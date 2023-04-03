import streamlit as st
from analisis_crediticio import *
import pandas as pd


st.set_page_config(page_title="Analisis crediticio con ML", layout="wide")
st.image('https://imageup.me/images/1bc21b54-ecb1-4fde-a7c6-582acb7cc4a8.png')

modelo_RF = cargar_modelo('ML_RFmodel4_Analisis_crediticio.pkl')
modelo_LR = cargar_modelo('ML_LRmodel2_Analisis_crediticio.pkl')

st.subheader('Ajuste en la prediccion')
option = ['Regresion logistica', 'Random forest']
# escoger el modelo preferido
model = st.selectbox('Que modelo prefiere usar?', option)

uploaded_file = st.file_uploader("Seleccione el archivo de prueba", type=['csv'], key="1")



if uploaded_file is not None:
    st.subheader('Evaluacion por Archivos')
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
    if st.button('RUN'):
        # Convierte el archivo cargado a un DataFrame
        if model == 'Regresion logistica':
            prediccion = predecir(modelo_LR, data)
            df_resultado = agregar_predicciones(data, prediccion)
            st.subheader('Evaluacion por Archivos')
            st.dataframe(df_resultado)
        else:
            prediccion = predecir(modelo_LR, data)
            df_resultado = agregar_predicciones(data, prediccion)
            st.subheader('Evaluacion por Archivos')
            st.dataframe(df_resultado)





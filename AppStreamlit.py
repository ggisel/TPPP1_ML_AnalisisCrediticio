#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importamos la libreria de streamlit para la interfaz de usuario
import streamlit as st
import pickle
import pandas as pd
from io import StringIO


# In[3]:


filename = 'ML_RFmodel4_Analisis_crediticio'

filename2 = 'ML_LRmodel2_Analisis_crediticio'
# In[4]:


modelo_RF4= pickle.load(open(filename,'rb'))
modelo_LR2= pickle.load(open(filename,'rb'))

# In[5]:


modelo_RF4.get_params()


# In[6]:


#Creamos la funcion para clasificar a los clientes segun lo que nos devuelva el modelo
def classify(num):
    if num == 0:
        return 'tipo 0'
    elif num == 1:
        return 'tipo 1'
    else:
        return 'tipo 2'


# In[1]:


def main():
    #titulo
    st.image('https://imageup.me/images/1bc21b54-ecb1-4fde-a7c6-582acb7cc4a8.png')
    
    #titulo de sidebar
    st.sidebar.header('Selección del Modelo')

    
    st.sidebar.subheader('')

    #escoger el modelo preferido
    option = ['Regresion logistica', 'Random forest']
    model = st.sidebar.selectbox('Que modelo prefiere usar?', option)

   
            
    st.subheader('Evaluar Archivo')
    
    uploaded_file = st.file_uploader("Seleccione el archivo de prueba")
    if uploaded_file is not None:
        # Convierte el archivo cargado a un DataFrame
        data2 = pd.read_csv(uploaded_file)
        # st.write(data)
        
        if model == 'Regresion logistica':
            
            predicciones= modelo_LR2.predict(data2)
        else:
            
            predicciones= modelo_RF4.predict(data2)

    nombres_valores = list(map(lambda x: "Denegar" if x == 0 else "Aceptar", predicciones))

    st.write(nombres_valores)    
       
       



# In[8]:


if __name__ == '__main__':
    main()


# In[ ]:





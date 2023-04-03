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
    st.sidebar.header('Si lo que necesitas es realizar una consulta individual, podes realizar una evaluacion estimativa brindando los siguientes datos:')
   
    
        
    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():

        st.sidebar.subheader('Ocupacion:')

        ocupacion = st.sidebar.radio(
            'Ocupacion:',
            ('Oficinista', 'profesional ejecutivo', 'area de ventas','emprendimientos','otro'))
        
        p1=p2=p3=p4=0
        if ocupacion == 'Oficinista':
            p1 = 1
        if ocupacion == 'profesional ejecutivo':
            p2 = 1
        if ocupacion == 'area de ventas':
            p3 = 1
        if ocupacion == 'emprendimientos':
            p4 = 1

        esPorDeuda = st.sidebar.radio(
            'es para saldar otra deuda?',
            ('Si','No'))

        if esPorDeuda == 'Si':
            p5 = 1
        
        p42 = st.sidebar.number_input('Antiguedad Laboral en años')

        

       

        
        p12 = st.sidebar.slider('Valor del prestamo', -1.0, 1.0)
        p22 = st.sidebar.slider('Saldo previo adeudado', -1.0, 1.0)
        p32 = st.sidebar.slider('Valor del inmueble hipotecado', -1.0, 1.0)
        
        p52 = st.sidebar.slider('Documentos derogados',0.0, 1.0)
        p13 = st.sidebar.slider('Delinq', 0.0, 1.0)
        p23 = st.sidebar.slider('Años de deuda mas antigua', -1.0, 1.0)
        p33 = st.sidebar.slider('ninq', -1.0, 1.0)
        p43 = st.sidebar.slider('clno', -1.0, 1.0)
        p53 = st.sidebar.slider('debtinc', -1.0, 1.0)

        #JOB_Office,JOB_ProfExe,JOB_Sales,JOB_Self,REASON_DebtCon,LOAN,MORTDUE,VALUE,YOJ,DEROG,DELINQ,CLAGE,NINQ,CLNO,DEBTINC
       
        data = {'JOB_Office': p1,
                'JOB_ProfExe': p2,
                'JOB_Sales': p3,
                'JOB_Self': p4,
                'REASON_DebtCon': p5,
                'LOAN': p12,
                'MORTDUE': p22,
                'VALUE': p32,
                'YOJ': p42,
                'DEROG': p52,
                'DELINQ': p13,
                'CLAGE': p23,
                'NINQ': p33,
                'CLNO': p43,
                'DEBTINC': p53,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()
    
    
    st.subheader('Ajuste en la prediccion')

    #escoger el modelo preferido
    option = ['Regresion logistica', 'Random forest']
    model = st.selectbox('Que modelo prefiere usar?', option)

   
    st.subheader('Vista previa de los parametros a procesar:')
    st.write(df)

    if st.button('RUN'):
        if model == 'Regresion logistica':
            
            st.success(classify(modelo_LR2.predict(df)))
        else:
            
            st.success(classify(modelo_RF4.predict(df)))
            
    st.subheader('Evaluacion por Archivos')
    
    uploaded_file = st.file_uploader("Seleccione el archivo de prueba")
    if uploaded_file is not None:
        # Convierte el archivo cargado a un DataFrame
        data2 = pd.read_csv(uploaded_file)
        # st.write(data)
        
        if model == 'Regresion logistica':
            
            st.success(modelo_LR2.predict(data2))
        else:
            
            st.success(modelo_RF4.predict(data2))



        #y_pred = modelo_RF4.predict(data)
        


        #st.subheader('predicciones generadas')
        # Muestra los resultados en Streamlit
        #st.write('Predictions:', y_pred)
       
       



# In[8]:


if __name__ == '__main__':
    main()


# In[ ]:





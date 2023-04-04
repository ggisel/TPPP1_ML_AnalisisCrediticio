# importamos la libreria de streamlit para la interfaz de usuario
import streamlit as st
import pickle
import pandas as pd

filename = 'ML_RFmodel4_Analisis_crediticio'

filename2='ML_LRmodel2_Analisis_crediticio'

modelo_RF4= pickle.load(open(filename,'rb'))
modelo_LR2= pickle.load(open(filename,'rb'))




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
        st.subheader('Resultados:')
        st.caption('0 si es viable otorgar el prestamo')
        st.caption('1 si debe ser negado ')
        if model == 'Regresion logistica':
            
            predicciones= modelo_LR2.predict(data2)
           
        else:
            
            predicciones= modelo_RF4.predict(data2)
        
        st.write(predicciones)
    




if __name__ == '__main__':
    main()





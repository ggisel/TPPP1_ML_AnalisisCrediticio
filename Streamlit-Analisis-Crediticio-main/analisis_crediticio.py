import pandas as pd
import pickle
def cargar_modelo(ruta):
    # Cargar el modelo entrenado desde el archivo pkl
    with open(ruta, 'rb') as pkl:
        archivo = pickle.load(pkl)
    return archivo

def predecir(modelo, datos):
    # Realizar la predicción con el modelo cargado
    prediccion = modelo.predict(datos)
    return prediccion
def clasificar(prediccion):
    if prediccion == 0:
        clasificacion = 'tipo 0'
    elif prediccion == 1:
        clasificacion = 'tipo 1'
    else:
        clasificacion = 'tipo 2'
    return clasificacion

def agregar_predicciones(dataframe, prediccion):
    df_pred = pd.DataFrame({'prediccion': prediccion})
    # concatenar el dataframe de predicciones con el dataframe original
    df_resultado = pd.concat([dataframe, df_pred], axis=1)
    df_resultado['clasificacion'] = df_resultado['prediccion'].apply(clasificar)
    df_resultado = df_resultado.drop('prediccion', axis=1)
    return df_resultado





import numpy as np
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model("calibrated_lightgbm")


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    label = predictions_df['Label'][0]
    prob = predictions_df['Score'][0] * 100
    return label, prob


## funcion para crear caracteristicas sobre el dataframe con variables minimas requeridas
def transform_df(input_df):
    ## porcentaje de ocupacion de los nodos
    input_df['tasa_ocupacion'] = input_df['HP OCUP'] / input_df['HP EFEC']
    ## tasa de crecimiento en rgus de un mes
    input_df['tasa_crecimiento_rgus_home'] = (
                (input_df['RGUS HOME'] - (input_df['RGUS HOME'] - input_df['NETO RGU'])) / (
                    input_df['RGUS HOME'] - input_df['NETO RGU']))
    ## valor promedio facturacion por home pass
    input_df['facturacion_promedio_hp'] = (input_df['FACTURACION'] / input_df['HP OCUP'])
    # Capacidad disponible del nodo
    input_df["capacidad_disponible_nodo"] = input_df["HP EFEC"] - input_df["HP OCUP"]
    # Promedio de servicios por hogar
    input_df["promedio_servicios_hogar"] = input_df["RGUS HOME"] / input_df["HP OCUP"]
    # porcentaje de servicios TO sobre total de servicios
    input_df['TASA_RGUS_TO_HOME'] = input_df['RGUS_TO_HOME'] / input_df['RGUS HOME']
    # porcentaje de servicios BA sobre total de servicios
    input_df['TASA_RGUS_BA_HOME'] = input_df['RGUS_BA_HOME'] / input_df['RGUS HOME']
    # porcentaje de servicios TV sobre total de servicios
    input_df['TASA_RGUS_TV_HOME'] = input_df['RGUS_TV_HOME'] / input_df['RGUS HOME']

    ## df para output
    df = input_df[['tasa_ocupacion', 'tasa_crecimiento_rgus_home',
                   'facturacion_promedio_hp', 'capacidad_disponible_nodo',
                   'promedio_servicios_hogar', 'TASA_RGUS_TO_HOME', 'TASA_RGUS_BA_HOME',
                   'TASA_RGUS_TV_HOME', 'ESTRATO_MODA', 'HP EFEC']]
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df


def run():
    add_selectbox = st.sidebar.selectbox("Tipo de predicción", ('Online', 'csv'))
    st.sidebar.info("Modelo para clasificar Nodos de acuerdo a su potencial comercial.")
    st.title('Clasificacion de nodos para area comercial.')

    if add_selectbox == 'Online':

        # 'tasa_ocupacion', 'tasa_crecimiento_rgus_home',
        # 'facturacion_promedio_hp', 'capacidad_disponible_nodo',
        # 'promedio_servicios_hogar', 'TASA_RGUS_TO_HOME', 'TASA_RGUS_BA_HOME',
        # 'TASA_RGUS_TV_HOME', 'ESTRATO_MODA','HP EFEC'

        HP_EFEC = st.number_input('Capacidad total en Homepasses del nodo', min_value=0)
        ESTRATO_MODA = st.selectbox('Estrato mas representativo en la ubicacion del nodo',
                                    ['ESTRATO1', 'ESTRATO2', 'ESTRATO3', 'ESTRATO4', 'ESTRATO5', 'ESTRATO6'])
        RGUS_HOME = st.number_input("RGUS Home", min_value=1)
        FACTURACION = st.number_input("Facturacion total del nodo en el mes.", min_value=0)
        HP_OCUP = st.number_input("Homepasses ocupados en el nodo.", min_value=0)
        NETO_RGU = st.number_input("Neto de conexiones para el nodo en el mes.", min_value=0)
        RGUS_TO_HOME = st.number_input("Total de RGUS Telefonia para el nodo.", min_value=0)
        RGUS_BA_HOME = st.number_input("Total de RGUS Banda Ancha para el nodo.", min_value=0)
        RGUS_TV_HOME = st.number_input("Total de RGUS Television para el nodo.", min_value=0)

        output = ""

        input_dict = {'FACTURACION': FACTURACION,
                      'HP EFEC': HP_EFEC,
                      'RGUS HOME': RGUS_HOME,
                      'NETO RGU': NETO_RGU,
                      'HP OCUP': HP_OCUP,
                      'RGUS_TO_HOME': RGUS_TO_HOME,
                      'RGUS_BA_HOME': RGUS_BA_HOME,
                      'RGUS_TV_HOME': RGUS_TV_HOME,
                      'ESTRATO_MODA': ESTRATO_MODA}

        input_df = pd.DataFrame([input_dict])
        df = transform_df(input_df)
        if st.button('Preddición'):
            label, prob = predict(model=model, input_df=df)
            st.success('La predicción es: {ou1}, con una certeza del {ou2}%'.format(ou1=str(label), ou2=str(prob)))
            predictions = predict_model(estimator=model, data=df)
            st.write(predictions)
    if add_selectbox == 'csv':
        st.subheader('Se requieren de manera obligatoria las siguientes columnas (los nombres deben coincidir exactamente)')
        file_upload = st.file_uploader('Subir Archivo CSV', type=['csv'])


        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)
            file= predictions.to_csv()
            st.download_button('Descargar Resultados',file,'text/csv')



if __name__ == '__main__':
    run()

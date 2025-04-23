import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Importar joblib para cargar el modelo

st.title("Proyecto Predicción precio de coches usados")
st.subheader("Análisis de datos y predicción de precios de coches usados")

st.info("Dataset: [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/data)")
st.markdown("El objetivo de esta competencia es predecir el precio de los autos usados ​​en función de varios atributos")

#cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_data_car.csv")
    return df

df = load_data()
st.dataframe(df.head())
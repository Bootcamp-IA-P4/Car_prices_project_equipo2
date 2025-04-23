import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Importar joblib para cargar el modelo

st.title("Modelo de predicción ")
st.subheader("Esta web muestra la predicción de precios de coches usados")
st.markdown("Este es un proyecto de predicción de precios de coches usados utilizando un modelo de regresión lineal. El modelo ha sido entrenado con datos de coches usados y se utiliza para predecir el precio de un coche en función de sus características.")   

st.info("Dataset: [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/data)")
st.markdown("El objetivo de esta competencia es predecir el precio de los autos usados ​​en función de varios atributos")

#cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_data_car.csv")
    return df

df = load_data()

# Importar el modelo entrenado
model = joblib.load("models/linear_regression_model.pkl")  # comprobar modelo y ruta


st.sidebar.header("Ingrese las características del coche")

# inputs
model_year = st.sidebar.slider("Año del modelo", min_value=1970, max_value=2025, value=2015, step=1)
milage = st.sidebar.number_input("Kilometraje (en millas)", min_value=0, max_value=500000, value=50000, step=1000)
transmission_num = st.sidebar.selectbox("Número de marchas", options=[1, 2, 4, 5, 6, 7, 8, 9, 10])


if st.sidebar.button("Predecir precio"):
    # Crea un DataFrame con las características ingresadas
    input_data = pd.DataFrame({
        "model_year": [model_year],
        "milage": [milage],
        "transmission_num": [transmission_num]
    })

    
    predicted_price = model.predict(input_data)[0]

    
    st.success(f"El precio estimado del coche es: ${predicted_price:,.2f}")
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib 
import json


json_path = "brand_model_mapping.json"
with open(json_path, 'r', encoding='utf-8') as json_file:
    brand_model_mapping = json.load(json_file)
##Importar el modelo entrenado y el label encoder
model = joblib.load("models/gradient_boosting_model.pkl")  # comprobar modelo y ruta
label_encoder = joblib.load("models/label_encoder.pkl")

st.sidebar.header("Ingrese las características del coche")

# inputs
brand = st.sidebar.selectbox("Marca del coche", options=list(brand_model_mapping.keys()))
if brand:
    model_car = st.sidebar.selectbox("Modelo del coche", options=brand_model_mapping[brand])

model_year = st.sidebar.slider("Año del modelo", min_value=1970, max_value=2025, value=2015, step=1)
milage = st.sidebar.number_input("Odómetro en millas", min_value=0, max_value=500000, value=50000, step=1000)
transmission_num = st.sidebar.selectbox("Número de marchas", options=[1, 2, 4, 5, 6, 7, 8, 9, 10])
accident = st.sidebar.selectbox("El coche ha sufrido algún accidente?", options=[0, 1])
engine_hp = st.sidebar.number_input("Caballos de fuerza del motor", min_value=0, max_value=1000, value=150, step=10)

if st.sidebar.button("Predecir precio"):
    enconded_brand = label_encoder.transform([brand])[0] 
    enconded_model_car = label_encoder.transform([model_car])[0] 
     # Codificar la marca
    # Crea un DataFrame con las características ingresadas
    input_data = pd.DataFrame({
        "brand": [enconded_brand],
        "model": [enconded_model_car],  
        "model_year": [model_year],
        "milage": [milage],
        "transmission_num": [transmission_num],
        "accident": [accident],
        "engine_hp": [0],
        })
   

    
    predicted_price = model.predict(input_data)[0]
    st.success(f"El precio estimado del coche es: ${predicted_price:,.2f}") ## redondea el resultado
    

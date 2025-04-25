import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Importar joblib para cargar el modelo


brands = ['MINI', 'Lincoln', 'Chevrolet', 'Genesis', 'Mercedes-Benz', 'Audi',
          'Ford', 'BMW', 'Tesla', 'Cadillac', 'Land', 'GMC', 'Toyota',
          'Hyundai', 'Volvo', 'Volkswagen', 'Buick', 'Rivian', 'RAM',
          'Hummer', 'Alfa', 'INFINITI', 'Jeep', 'Porsche', 'McLaren',
          'Honda', 'Lexus', 'Dodge', 'Nissan', 'Jaguar', 'Acura', 'Kia',
          'Mitsubishi', 'Rolls-Royce', 'Maserati', 'Pontiac', 'Saturn',
          'Bentley', 'Mazda', 'Subaru', 'Ferrari', 'Aston', 'Lamborghini',
          'Chrysler', 'Lucid', 'Lotus', 'Scion', 'smart', 'Karma',
          'Plymouth', 'Suzuki', 'FIAT', 'Saab', 'Bugatti', 'Mercury',
          'Polestar', 'Maybach']

##Importar el modelo entrenado y el label encoder
model = joblib.load("models/gradient_boosting_model.pkl")  # comprobar modelo y ruta
label_encoder = joblib.load("models/label_encoder.pkl")

st.sidebar.header("Ingrese las características del coche")

# inputs
brand = st.sidebar.selectbox("Marca", options=brands)
# model_car = st.sidebar.text_input("Modelo del coche", value="")
model_year = st.sidebar.slider("Año del modelo", min_value=1970, max_value=2025, value=2015, step=1)
milage = st.sidebar.number_input("Odómetro en millas", min_value=0, max_value=500000, value=50000, step=1000)
transmission_num = st.sidebar.selectbox("Número de marchas", options=[1, 2, 4, 5, 6, 7, 8, 9, 10])



if st.sidebar.button("Predecir precio"):
    enconded_brand = label_encoder.transform([brand])[0] 
     # Codificar la marca
    # Crea un DataFrame con las características ingresadas
    input_data = pd.DataFrame({
        "brand": [enconded_brand],
        # "model": [model_car],  
        "model_year": [model_year],
        "milage": [milage],
        "transmission_num": [transmission_num]
    })

    
    predicted_price = model.predict(input_data)[0]
    st.success(f"El precio estimado del coche es: ${predicted_price:,.2f}") ## redondea el resultado
    

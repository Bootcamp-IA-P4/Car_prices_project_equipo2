from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import pandas as pd
import os
# from pipeline import create_pipeline

# Setting paths
current_dir = os.getcwd()  # Use os.getcwd() to get the current working directory
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data")
file_path = os.path.join(data_dir, "")
cars_csv_path = os.path.join(data_dir, "raw_data.csv")
clean_data_car_csv_path = os.path.join(data_dir, "clean_data_car.csv")
train_csv_path = os.path.join(data_dir, "train_csv.csv")
test_csv_path = os.path.join(data_dir, "test_csv.csv")
# Load csv and drop NaN values
df = pd.read_csv(clean_data_car_csv_path)
df = df.dropna()

# ##  dividir datos en conjuntos de entrenamiento y prueba.
# def split_data(df, features, target, test_size=0.2, random_state=42):
#     X_train, X_test, y_train, y_test = train_test_split(
#         df[features],
#         df[target],
#         test_size=test_size,
#         random_state=random_state
#     )
#     return X_train, X_test, y_train, y_test

# # Valoración del modelo y detección de overfitting
# # Asegúrate de que tienes instaladas las librerías necesarias

# Si tienes XGBoost instalado:
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# Preprocesado de la variable brand
if 'brand' in df.columns:
    print('Preprocesando la variable brand...')
    
    # Codificar la columna 'brand' con LabelEncoder
    label_encoder = LabelEncoder()
    df['brand'] = label_encoder.fit_transform(df['brand'])

    # Guardar el LabelEncoder para usarlo en predicciones futuras
    joblib.dump(label_encoder, "../models/label_encoder.pkl")
    print("LabelEncoder guardado en '../models/label_encoder.pkl'.")

    # Seleccionar características y objetivo
    features = ['brand', 'model_year', 'milage', 'transmission_num']
    X = df[features]
    y = df['price']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Agregar la columna 'price' a los conjuntos de datos antes de guardarlos
    X_train['price'] = y_train
    X_test['price'] = y_test

    # Guardar los conjuntos de datos en archivos CSV
    X_train.to_csv(train_csv_path, index=False)
    X_test.to_csv(test_csv_path, index=False)
    print(f"Datos preprocesados guardados en {train_csv_path} y {test_csv_path}.")
else:
    print("La columna 'brand' no está presente en el dataset. Verifica el archivo 'clean_data_car.csv'.")


# Variables predictoras y objetivo
features = ['brand', 'model_year', 'milage', 'transmission_num']
X_train = pd.read_csv(train_csv_path)
X_test = pd.read_csv(test_csv_path)

# Separar la variable objetivo
y_train = X_train.pop('price')
y_test = X_test.pop('price')

print("Primeras filas de X_train:")
print(X_train.head())

print("Primeras filas de X_test:")
print(X_test.head())

# Lista de modelos 
models = {
"Gradient Boosting":GradientBoostingRegressor(
    n_estimators=1000,         # Aumentar si learning_rate es bajo
    learning_rate=0.03,       # Más bajo = más robusto
    max_depth=3,              # Controla complejidad. Prueba 3–5
    min_samples_split=5,      # Evita divisiones pequeñas
    min_samples_leaf=10,       # Evita hojas con pocos datos
    subsample=0.7,            # Usa solo parte del dataset por árbol
    max_features='sqrt',      # Menos features = menos varianza
    random_state=42
),

    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor()
}
results = []
if xgb_available:
    models["XGBoost"] = XGBRegressor(random_state=42)

# Evaluación de modelos con detección de overfitting
results = []

for name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Hacer predicciones en entrenamiento y prueba
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas en entrenamiento
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    # Calcular métricas en prueba
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)

    # Detectar overfitting de forma sencilla (comparando R2)
    overfitting = False
    if r2_test < r2_train - 0.05:  # Una diferencia del 5% en R2 podría indicar overfitting
        overfitting = True

    results.append({
        'Model': name,
        'R2 Train': r2_train,
        'RMSE Train': rmse_train,
        'MAE Train': mae_train,
        'R2 Test': r2_test,
        'RMSE Test': rmse_test,
        'MAE Test': mae_test,
        'Overfitting': overfitting,
        'Overfitting %': 100 * (r2_train - r2_test) / abs(r2_train) if r2_train != 0 else None,

    })

# Mostrar resultados ordenados
results_df = pd.DataFrame(results).sort_values(by='R2 Test', ascending=False)
print("\n Comparación de modelos con detección de Overfitting:\n")
print(results_df)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression,  Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor

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


# Preprocesado de las variables categóricas
if 'brand' in df.columns and 'model' in df.columns:
    # Verificar si las columnas 'brand' y 'model' existen
    print('Preprocesando la variables categóricas...')
    # Codificar la columna 'brand' con LabelEncoder
    label_encoder = LabelEncoder()
    df['brand'] = label_encoder.fit_transform(df['brand'])
    df['model'] = label_encoder.fit_transform(df['model'])
    # Guardar el LabelEncoder para usarlo en predicciones futuras
    joblib.dump(label_encoder, "../models/label_encoder.pkl")
    print("LabelEncoder guardado en '../models/label_encoder.pkl'.")

# Seleccionar características y objetivo
features = ['brand','model', 'model_year', 'milage', 'engine_hp','engine_cylinder','transmission_num','accident',]
X = df[features]
y = df['price']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista de modelos 
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting":GradientBoostingRegressor(
    n_estimators=1500,         # Aumentar si learning_rate es bajo
    learning_rate=0.02,       # Más bajo = más robusto
    max_depth=3,              # Controla complejidad. Prueba 3–5
    min_samples_split=5,      # Evita divisiones pequeñas
    min_samples_leaf=5,       # Evita hojas con pocos datos
    subsample=0.8,            # Usa solo parte del dataset por árbol
    max_features='sqrt',      # Menos features = menos varianza
    random_state=42
),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),  
    "LassoCV": LassoCV(cv=5, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
    n_estimators=100,         # Reduce o aumenta según el tamaño del dataset
    max_depth=20,             # Controla la profundidad máxima de los árboles
    min_samples_split=5,      # Aumenta para evitar divisiones con muy pocos datos
    min_samples_leaf=2,       # Aumenta para reducir overfitting
    max_features='sqrt',      # Usa sqrt o log2 para limitar el nº de features por árbol
    random_state=42
    )
}

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
        'R2 Train': round(r2_train,2),
        'RMSE Train': round(rmse_train,2),
        'MAE Train': round(mae_train,2),
        'R2 Test': round(r2_test,2),
        'RMSE Test': round(rmse_test,2),
        'MAE Test': round(mae_test,2),
        'Overfitting': overfitting,
        'Overfitting %': round((100 * (r2_train - r2_test) / abs(r2_train)),2) if r2_train != 0 else None,

    })

# Mostrar resultados ordenados
results_df = pd.DataFrame(results).sort_values(by='R2 Test', ascending=False)
print("\n Comparación de modelos con detección de Overfitting:\n")
print(results_df)



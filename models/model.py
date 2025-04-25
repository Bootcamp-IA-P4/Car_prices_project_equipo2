from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression,  Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import os

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


##  dividir datos en conjuntos de entrenamiento y prueba.
def split_data(df, features, target, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[target],
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test



# Valoración del modelo y detección de overfitting
# Asegúrate de que tienes instaladas las librerías necesarias

# Si tienes XGBoost instalado:
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False



# Variables predictoras y objetivo
features = ['model_year', 'milage', 'engine_hp','accident','transmission_num','engine_cylinder',]
X = df[features]
y = df['price']

# División del dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista de modelos 
models = {
    # "Linear Regression": LinearRegression(),
    
    "Gradient Boosting":GradientBoostingRegressor(
    n_estimators=1000,         # Aumentar si learning_rate es bajo
    learning_rate=0.02,       # Más bajo = más robusto
    max_depth=3,              # Controla complejidad. Prueba 3–5
    min_samples_split=5,      # Evita divisiones pequeñas
    min_samples_leaf=5,       # Evita hojas con pocos datos
    subsample=0.8,            # Usa solo parte del dataset por árbol
    max_features='sqrt',      # Menos features = menos varianza
    random_state=42
),

    # "K-Nearest Neighbors": KNeighborsRegressor(),
    # "Ridge Regression": Ridge(random_state=42),
    # "Lasso Regression": Lasso(random_state=42),  
    # "LassoCV": LassoCV(cv=5, random_state=42),
    # "XGBoost": XGBRegressor(random_state=42),
    # "Random Forest": RandomForestRegressor(
    # n_estimators=100,         # Reduce o aumenta según el tamaño del dataset
    # max_depth=20,             # Controla la profundidad máxima de los árboles
    # min_samples_split=5,      # Aumenta para evitar divisiones con muy pocos datos
    # min_samples_leaf=2,       # Aumenta para reducir overfitting
    # max_features='sqrt',      # Usa sqrt o log2 para limitar el nº de features por árbol
    # random_state=42
    # )
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
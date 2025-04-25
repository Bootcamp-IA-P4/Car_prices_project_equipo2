from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# Setting paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data")
cars_csv_path = os.path.join(data_dir, "raw_data.csv")
clean_data_car_csv_path = os.path.join(data_dir, "clean_data_car.csv")

# Load csv and drop NaN values if any
df = pd.read_csv(clean_data_car_csv_path)
df = df.dropna()

# Preprocesado de las variables categóricas
if 'brand' in df.columns and 'model' in df.columns:
    label_encoder = LabelEncoder()
    df['brand'] = label_encoder.fit_transform(df['brand'])
    df['model'] = label_encoder.fit_transform(df['model'])
    joblib.dump(label_encoder, "../models/label_encoder.pkl")

# Seleccionar características y objetivo
features = ['brand','model', 'model_year', 'milage', 'engine_hp','engine_cylinder','transmission_num','accident']
X = df[features]
y = df['price']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- RANDOMIZED SEARCH PARA HYPERPARAMETERS ---
param_grid = {
    'n_estimators': [300, 500, 1000, 1500],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'subsample': [0.7, 0.8, 1.0],
    'max_features': ['sqrt', 'log2']
}

base_model = GradientBoostingRegressor(random_state=42)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Entrenamos el modelo con búsqueda de hiperparámetros
search.fit(X_train, y_train)
gradient_boosting = search.best_estimator_

print("Mejores hiperparámetros encontrados:")
print(search.best_params_)

# --- PREDICCIONES Y MÉTRICAS ---
y_train_predict = gradient_boosting.predict(X_train)
y_test_predict = gradient_boosting.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_predict)
r2_train = r2_score(y_train, y_train_predict)
mae_train = mean_absolute_error(y_train, y_train_predict)
rmse_train = np.sqrt(mse_train)

mse_test = mean_squared_error(y_test, y_test_predict)
r2_test = r2_score(y_test, y_test_predict)
mae_test = mean_absolute_error(y_test, y_test_predict)
rmse_test = np.sqrt(mse_test)

overfitting = False
overfitting_threshold = 0.05
if r2_test < r2_train - overfitting_threshold:
    overfitting = True
overfitting_percentage = 100 * (r2_train - r2_test) / abs(r2_train) if r2_train != 0 else 0

print("Gradient Boosting Regressor (Optimizado)")
print(f"R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")
print(f"RMSE Train: {rmse_train:.2f}, RMSE Test: {rmse_test:.2f}")
print(f"MAE Train: {mae_train:.2f}, MAE Test: {mae_test:.2f}")
print(f"Overfitting: {overfitting_percentage:.2f}%")
print(f"Overfitting detected: {overfitting_percentage:.2f}%" if overfitting else "No overfitting detected")

# Guardar el mejor modelo
# joblib.dump(gradient_boosting, "../models/gradient_boosting_model.pkl")


# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# Mejores hiperparámetros encontrados:
# {'subsample': 0.8, 'n_estimators': 1500, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2', 'max_depth': 5, 'learning_rate': 0.02}
# Gradient Boosting Regressor (Optimizado)
# R2 Train: 0.4631, R2 Test: 0.4005
# RMSE Train: 28549.57, RMSE Test: 29985.96
# MAE Train: 13184.65, MAE Test: 13771.40
# Overfitting: 13.51%
# Overfitting detected: 13.51%
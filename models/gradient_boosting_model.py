from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib

## primero cargamos los datos:
train_csv_path = "../data/train_csv.csv"
test_csv_path = "../data/test_csv.csv"

X_train = pd.read_csv(train_csv_path)
X_test = pd.read_csv(test_csv_path)

y_train = X_train.pop('price')
y_test = X_test.pop('price')

## vamos a crear modelo 

gradient_boosting = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=10,
    subsample=0.7,
    max_features='sqrt',
    random_state=42
)

## entrenar el modelo
gradient_boosting.fit(X_train, y_train)

## Hacer las predicciones
y_train_predict = gradient_boosting.predict(X_train)
y_test_predict = gradient_boosting.predict(X_test)

## Métricas entrnamiento y prueba:
mse_train = mean_squared_error(y_train, y_train_predict)
r2_train = r2_score(y_train, y_train_predict)
mae_train = mean_absolute_error(y_train, y_train_predict)
rmse_train = np.sqrt(mse_train)

mse_test = mean_squared_error(y_test, y_test_predict)
r2_test = r2_score(y_test, y_test_predict)
mae_test = mean_absolute_error(y_test, y_test_predict)
rmse_test = np.sqrt(mse_test)

overfitting = False
overfitting_threshold = 0.05  # Diferencia del 5% en R2
if r2_test < r2_train - overfitting_threshold:
    overfitting = True
overfitting_percentage = 100 * (r2_train - r2_test) / abs(r2_train) if r2_train != 0 else 0
## resultados:

print("Gradient Boosting Regressor")
print(f"R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")
print(f"RMSE Train: {rmse_train:.2f}, RMSE Test: {rmse_test:.2f}")
print(f"MAE Train: {mae_train:.2f}, MAE Test: {mae_test:.2f}")
print(f"Overfitting: {overfitting_percentage:.2f}%")
print(f"Overfitting detected: {overfitting_percentage:.2f}%" if overfitting else "No overfitting detected")


## Vamos a guardar el modelo entrenado
joblib.dump(gradient_boosting, "../models/gradient_boosting_model.pkl")
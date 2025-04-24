from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


##  dividir datos en conjuntos de entrenamiento y prueba.
def split_data(df, features, target, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[target],
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


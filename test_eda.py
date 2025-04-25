def get_relationship_correlations(df_car, target, predictors):
    correlations = {}
    for predictor in predictors:
        if predictor in df_car.columns:
            correlation = df_car[target].corr(df_car[predictor])
            correlations[predictor] = correlation
    return correlations


import pandas as pd
import numpy as np
from test_eda import get_relationship_correlations  

def test_get_relationship_correlations_correctness():
    # Datos de prueba
    df = pd.DataFrame({
        'price': [10000, 15000, 20000, 25000],
        'engine': [1000, 1500, 2000, 2500]
    })
    
    result = get_relationship_correlations(df, 'price', ['engine'])
    
    expected_corr = df['price'].corr(df['engine'])
    assert 'engine' in result
    assert np.isclose(result['engine'], expected_corr), "La correlación calculada no es correcta"

def test_get_relationship_correlations_missing_column():
    df = pd.DataFrame({
        'price': [10000, 15000, 20000],
        'mileage': [10, 20, 30]
    })
    
    result = get_relationship_correlations(df, 'price', ['engine'])  # columna no existe
    assert result == {}, "Debe devolver un diccionario vacío si las columnas no existen"


def test_get_relationship_correlations_with_nan():
    df = pd.DataFrame({
        'price': [10000, 15000, np.nan, 25000],
        'engine': [1000, 1500, 2000, np.nan]
    })
    
    result = get_relationship_correlations(df, 'price', ['engine'])
    
    # Calculamos la correlación esperada usando pandas, que ignora NaNs automáticamente
    expected_corr = df['price'].corr(df['engine'])
    
    assert 'engine' in result
    assert np.isclose(result['engine'], expected_corr, equal_nan=True), "La correlación con NaNs no es correcta"


def test_get_relationship_correlations_empty_df():
    df = pd.DataFrame(columns=['price', 'engine'])  # columnas definidas pero sin datos
    
    result = get_relationship_correlations(df, 'price', ['engine'])
    
    assert result == {}, "Debe devolver un diccionario vacío con un DataFrame vacío"

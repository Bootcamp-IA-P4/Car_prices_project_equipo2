import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def load_pipeline():
    pipeline = joblib.load("models/model_pipeline.pkl")  
    return pipeline
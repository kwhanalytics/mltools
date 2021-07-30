"""
Module for storing shared constants
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler


DEFAULT_METRICS = ["rmse", "r2"]

# Lookup for sklearn estimators given string names for yaml functionality
MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "LinearRegression": LinearRegression
}

# Constant random state if specified
RANDOM_STATE = np.random.RandomState(seed=42)

# Lookup for sklearn scalers given string names for yaml functionality
SCALERS = {
    "StandardScaler": StandardScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "RobustScaler": RobustScaler
}



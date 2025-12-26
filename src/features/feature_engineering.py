import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import yaml

class FeatureEngineer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.numeric_features = self.config['data']['numeric_features']
        self.categorical_features = self.config['data']['categorical_features']
        self.target = self.config['data']['target']
        
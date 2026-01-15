import pandas as pd
import numpy as np
from feast import FeatureStore
from datetime import datetime, timedelta
import os

class FeastFeatureManager:
    
    def __init__(self, repo_path = "feature_store"):
        self.repo_path = repo_path
        self.store = None
        
        if os.path.exists(repo_path):
            self.store = FeatureStore(repo_path = repo_path)
            
    def setup_feature_store(self):
        print("\n" + "=" * 60)
        print("SETTING UP FEAST FEATURE STORE")
        print("=" * 60)
        
        os.makedirs(self.repo_path, exist_ok = True)
        os.makedirs(f"{self.repo_path}/data", exist_ok = True)
        
        config = """
        project: churn_prediction
        registry: data/registry.db
        provider: local
        online_store:
            type: sqlite
            path: data/onlince_store.db
        """
        
        with open(f"{self.repo_path}/feature_store.yaml", 'w') as f:
            f.write(config)
            
        print("Created feature_store.yaml")
        print("Feature store initialized")
        print(f"Location: {self.repo_path}")
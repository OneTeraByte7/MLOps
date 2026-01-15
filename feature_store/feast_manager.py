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
        
    
    def materialize_features(self, df: pd.DataFrame,
                             start_date: datetime = None,
                             end_date: datetime = None):
        
        if self.store is None:
            raise RuntimeError("feature store not initialized")
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days = 7)
            
        if end_date is None:
            end_date = datetime.now()
            
        print("\n" + "=" * 60)
        print("Materializing features to features store")
        print("=" * 60)
        
        if 'event_timestamp' not in df.columns:
            df['event_timestamp'] = datetime.now()
            
        output_path = f"{self.repo_path}/data/customer_stats.parquet"
        df.to_parquet(output_path, index = False)
        
        print(f"Saved batch features to {output_path}")
        print(f"Records: {len(df)}")
        
        try:
            self.store.materialize(
                start_date = start_date,
                end_date = end_date
            )
            
            print("Features materialized to online store")
            
        except Exception as e:
            print(f"Materialization error: {e}")
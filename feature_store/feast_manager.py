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
            
            
    def get_online_features(self, customer_ids: list) -> pd.DataFrame:
        
        if self.store is None:
            raise RuntimeError("Feature store not initialized")
        
        features = [
            "customer_engagement: logins_per_month",
            "customer_engagement: feature_usage_depth",
            "customer_engagement: api_calls_per_month",
            "customer_engagement: days_since_last_login",
            "customer_support: support_tickest",
            "customer_support: avg_ticket_resolution_days",
            "customer_support: nps_score",
            "customer_billing: monthly_revenue",
            "customer_billing: payment_delays",
            "customer_billing: contract_length_months",
            "customer_profile: account_age_days",
            "customer_profile: subscription_tier",
            "customer_profile: team_size",
        ]
        
        entity_df = pd.DataFrame({
            "customer_id": customer_ids
        })
        
        feature_vector = self.store.get_onlince_features(
            features = features,
            entity_rows = entity_df.to_dict('records')
        )
        
        features_df = pd.DataFrame(feature_vector.to_dict())
        
        return features_df
    
    def get_historical_features(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        
        if self.store is None:
            raise RuntimeError("Feature store not initialized")
        
        features = [
            "customer_engagement: logins_per_month",
            "customer_engagement: feature_usage_depth",
            "customer_engagement: api_calls_per_month",
            "customer_engagement: days_since_last_login",
            "customer_support: support_tickets",
            "customer_support: avg_ticket_resolution_days",
            "customer_support: nps_score",
            "customer_billing: monthly_revenue",
            "customer_billing: payment_delays",
            "customer_billing: contract_length_months",
            "customer_profile: account_age_days",
            "customer_profile: subscription_tier",
            "customer_profile: team_size",
        ]
        
        training_df = self.store.get_historical_features(
            entity_df = entity_df,
            features = features
        ).to_df()
        
        return training_df
    
    def validate_features(self):
        if self.store is None:
            raise RuntimeError("Features store not initialized")
        
        print("\n" + "=" * 60)
        print("Validating features")
        print("=" * 60)
        
        feature_views = self.store.list_feature_views()
        
        print(f"\n Feature Views ({len(feature_views)}):")
        for fv in feature_views:
            print(f"{fv.name}")
            print(f"Features: {[f.name for f in fv.schema]}")
            print(f" TTL: {fv.ttl}")
            
        entities = self.store.list_entities()
        print(f"\n Entities ({len(entities)}):")
        for entity in entities:
            print(f" {entity.name}")
            
        print("\n" + "=" * 60)
        print("Feature Store Validation Complete")
        print("=" * 60)
        
    def get_feature_statistics(self, feature_view_name: str) -> pd.DataFrame:
        if self.store is None:
            raise RuntimeError("Feature store not initialized")
        
        print(f"Statistics for {feature_view_name} (implement based on your needs)")
        return pd.DataFrame()
    

if __name__ == "__main__":
    fs_manager = FeastFeatureManager()
    
    print("\n Example 1: Materializing Features")
    print("=" * 40)
    
    train_df = pd.read_csv("data/raw/train.csv")
    train_df['event_timestamp'] = datetime.now()
    
    print("\n Example 2: Onlince Feature Retrieval")
    print("-" * 40)
    
    customer_ids = ["CUST_000001", "CUST_000002", "CUST_000003"]
    
    print("\n Example 3: Historical Feature Retrieval (point-in-Time Correct)")
    print("-" * 40)
    
    entity_df = pd.DataFrame({
        'customer_id': ["CUST_000001", "CUST_000002"],
        'event_timestamp': [
            datetime.now() - timedelta(days = 30),
            datetime.now() - timedelta(days = 30)
        ]
    })
    
    print("\n" + "=" * 60)
    print("Feature Store Benefits: ")
    print("=" * 60)
    print("Point-in-Time correct features (no data leakage)")
    print("Low Latency online serving")
    print("Feature reuse across models")
    print("Consistent train/serve")
    print("Feature versioning")
    print("Feature discovery")
    print("=" * 60)
    

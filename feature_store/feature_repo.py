from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

customer = Entity(
    name = "customer",
    join_keys = ["customer_id"],
    description = "Customer entity for churn prediction"
)

customer_stats_source = FileSource(
    path = "data/feature_store_customer_stats.parquet",
    timestamp_filed = "event_timestamp",
)

customer_engagement_feeatures = FeatureView(
    name = "customer_engagement_features",
    entities = [customer],
    ttl = timedelta(days = 1),
    schema = [
        Field(name = "logins_per_month", dtype = Int64),
        Field(name = "feature_usage_depth", dtype = Float32),
        Field(name = "api_calls_per_month", dtype = Int64),
        Field(name = "days_since__last_login", dtype = Int64),
    ],
    online = True,
    source = customer_stats_source,
    tags = {"team": "ml", "category": "engagement" }
)


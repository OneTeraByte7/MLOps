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

customer_support_features = FeatureView(
    name = "customer_support",
    entities = [customer],
    ttl = timedelta(days=1),
    schema = [
        Field(name = "support_tickets", dtype = Int64),
        Field(name = "avg_ticket_resolution_days", dtype = Float32),
        Field(name = "nps_score", dtype = Int64),
    ],
    online = True,
    source = customer_stats_source,
    tags = {"team": "ml", "category": "support"}
)

customer_billing_features = FeatureView(
    name = "customer_billing",
    entites = [customer],
    ttl = timedelta(days=1),
    schema = [
        Field(name = "monthly_revenue", dtype = Float32),
        Field(name = "payment_delays", dtype = Int64),
        Field(name = "contract_length_months", dtype = Int64),
    ],
    online = True,
    source = customer_stats_source,
    tags = {"team": "ml", "category": "billing"}
)

customer_profile_features = FeatureView(
    name = "customer_profile",
    entities = [customer],
    ttl = [customer],
    ttl = timedelta(days = 30),
    schema = [
        Field(name = "account_age_days", dtype = Int64),
        Field(name = "subscription_tier", dtype = String),
        Field(name = "team_size", dtype = Int64),
    ],
    onlince = True,
    source = customer_stats_source,
    tags = {"team": "ml", "category": "profile"}
)




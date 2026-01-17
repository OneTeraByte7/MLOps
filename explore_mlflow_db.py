"""
Query MLflow SQLite database and show available data
"""
import sqlite3
import pandas as pd
from datetime import datetime

def explore_mlflow_db():
    """Explore what's in the MLflow database"""
    conn = sqlite3.connect('mlflow.db')
    
    print("=" * 70)
    print("MLflow Database Exploration")
    print("=" * 70)
    
    # Get all tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print("\n📊 Available Tables:")
    for table in tables['name']:
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'][0]
        print(f"  • {table}: {count} rows")
    
    # Check experiments
    print("\n" + "=" * 70)
    print("Experiments:")
    print("=" * 70)
    experiments = pd.read_sql("SELECT * FROM experiments", conn)
    if len(experiments) > 0:
        print(experiments[['experiment_id', 'name', 'lifecycle_stage']])
    else:
        print("  No experiments found")
    
    # Check runs
    print("\n" + "=" * 70)
    print("Runs:")
    print("=" * 70)
    runs = pd.read_sql("""
        SELECT run_uuid, experiment_id, status, 
               datetime(start_time/1000, 'unixepoch') as start_time,
               datetime(end_time/1000, 'unixepoch') as end_time
        FROM runs 
        ORDER BY start_time DESC 
        LIMIT 10
    """, conn)
    if len(runs) > 0:
        print(f"  Total runs: {len(runs)}")
        print(runs)
    else:
        print("  No runs found")
    
    # Check metrics with actual values
    print("\n" + "=" * 70)
    print("Available Metrics:")
    print("=" * 70)
    metrics_summary = pd.read_sql("""
        SELECT key, COUNT(*) as count, 
               ROUND(AVG(value), 4) as avg_value,
               ROUND(MIN(value), 4) as min_value,
               ROUND(MAX(value), 4) as max_value
        FROM metrics 
        GROUP BY key
    """, conn)
    if len(metrics_summary) > 0:
        print(metrics_summary)
    else:
        print("  No metrics found")
    
    # Get recent metrics with run info
    print("\n" + "=" * 70)
    print("Recent Metric Values (Sample):")
    print("=" * 70)
    recent_metrics = pd.read_sql("""
        SELECT m.run_uuid, m.key, m.value, m.step,
               datetime(m.timestamp/1000, 'unixepoch') as timestamp
        FROM metrics m
        ORDER BY m.timestamp DESC
        LIMIT 20
    """, conn)
    if len(recent_metrics) > 0:
        print(recent_metrics)
    else:
        print("  No metrics found")
    
    # Check parameters
    print("\n" + "=" * 70)
    print("Parameters:")
    print("=" * 70)
    params_summary = pd.read_sql("""
        SELECT key, COUNT(DISTINCT value) as unique_values, 
               COUNT(*) as total_runs
        FROM params 
        GROUP BY key
    """, conn)
    if len(params_summary) > 0:
        print(params_summary)
    else:
        print("  No parameters found")
    
    # Get sample params
    print("\n" + "=" * 70)
    print("Sample Parameter Values:")
    print("=" * 70)
    sample_params = pd.read_sql("""
        SELECT run_uuid, key, value
        FROM params
        LIMIT 20
    """, conn)
    if len(sample_params) > 0:
        print(sample_params)
    else:
        print("  No parameters found")
    
    conn.close()
    
    return len(runs) > 0

if __name__ == "__main__":
    has_data = explore_mlflow_db()
    
    print("\n" + "=" * 70)
    if has_data:
        print("✅ Database has training data!")
        print("   The dashboard can now use this real data.")
    else:
        print("⚠️  Database is empty. Run: python train_model.py")
    print("=" * 70)

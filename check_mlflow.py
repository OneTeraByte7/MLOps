"""
Check MLflow database and extract training data
"""
import sqlite3
import pandas as pd
import json
from datetime import datetime

def check_mlflow_db():
    """Check what's in the MLflow database"""
    try:
        conn = sqlite3.connect('mlflow.db')
        
        # Check tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("=" * 60)
        print("MLflow Database Tables:")
        print("=" * 60)
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check experiments
        print("\n" + "=" * 60)
        print("Experiments:")
        print("=" * 60)
        experiments = pd.read_sql("SELECT * FROM experiments", conn)
        print(experiments)
        
        # Check runs
        print("\n" + "=" * 60)
        print("Runs:")
        print("=" * 60)
        runs = pd.read_sql("SELECT run_uuid, experiment_id, status, start_time, end_time FROM runs", conn)
        print(f"Total runs: {len(runs)}")
        print(runs.head())
        
        # Check metrics
        print("\n" + "=" * 60)
        print("Metrics:")
        print("=" * 60)
        metrics = pd.read_sql("SELECT * FROM metrics LIMIT 20", conn)
        print(metrics)
        
        # Check params
        print("\n" + "=" * 60)
        print("Parameters:")
        print("=" * 60)
        params = pd.read_sql("SELECT * FROM params LIMIT 20", conn)
        print(params)
        
        conn.close()
        
        return {
            'experiments': len(experiments),
            'runs': len(runs),
            'has_data': len(runs) > 0
        }
        
    except Exception as e:
        print(f"Error reading MLflow database: {e}")
        return None

if __name__ == "__main__":
    result = check_mlflow_db()
    
    if result and result['has_data']:
        print("\n" + "=" * 60)
        print("✓ MLflow database has training data!")
        print(f"  Experiments: {result['experiments']}")
        print(f"  Runs: {result['runs']}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠ No data found in MLflow database")
        print("  Run: python train_model.py")
        print("=" * 60)

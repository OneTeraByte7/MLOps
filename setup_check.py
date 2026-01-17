"""
Complete setup script - Check database, start services, populate dashboard
"""
import subprocess
import sys
import os
import sqlite3
import pandas as pd

def check_mlflow_database():
    """Check if MLflow database has training data"""
    print("=" * 70)
    print("Step 1: Checking MLflow Database")
    print("=" * 70)
    
    if not os.path.exists('mlflow.db'):
        print("✗ mlflow.db not found")
        print("  Run: python train_model.py")
        return False
    
    try:
        conn = sqlite3.connect('mlflow.db')
        runs_count = pd.read_sql("SELECT COUNT(*) as count FROM runs", conn)['count'][0]
        metrics_count = pd.read_sql("SELECT COUNT(*) as count FROM metrics", conn)['count'][0]
        conn.close()
        
        print(f"✓ Found {runs_count} training runs")
        print(f"✓ Found {metrics_count} metrics")
        
        if runs_count > 0:
            print("✓ Database has training data - Dashboard will show real metrics!")
            return True
        else:
            print("⚠ Database is empty - Train a model first")
            return False
            
    except Exception as e:
        print(f"✗ Error reading database: {e}")
        return False

def check_training_data():
    """Check if training data exists"""
    print("\n" + "=" * 70)
    print("Step 2: Checking Training Data")
    print("=" * 70)
    
    data_paths = [
        'data/test_data.csv',
        'data/train_data.csv',
        'data/raw/churn_data.csv'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ Found {path}")
            return True
    
    print("⚠ No training data found")
    print("  Run: python data_generator.py")
    return False

def check_model_files():
    """Check if trained model exists"""
    print("\n" + "=" * 70)
    print("Step 3: Checking Model Files")
    print("=" * 70)
    
    model_files = [
        'models/model.json',
        'models/preprocessor.pkl'
    ]
    
    all_exist = True
    for path in model_files:
        if os.path.exists(path):
            print(f"✓ Found {path}")
        else:
            print(f"✗ Missing {path}")
            all_exist = False
    
    if all_exist:
        print("✓ Model files ready - API can make predictions!")
        return True
    else:
        print("⚠ Model files missing")
        print("  Run: python train_model.py")
        return False

def main():
    print("\n" + "=" * 70)
    print("MLOps System Setup & Health Check")
    print("=" * 70 + "\n")
    
    has_db_data = check_mlflow_database()
    has_training_data = check_training_data()
    has_model = check_model_files()
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    if has_db_data and has_model:
        print("✅ System is ready!")
        print("\nYou can now:")
        print("  1. Start services: python start_services.py")
        print("  2. Open dashboard: http://localhost:5173")
        print("  3. Dashboard will show REAL data from SQLite database")
        print("\nOptional:")
        print("  • Make new predictions: python populate_dashboard.py")
        
    elif has_model:
        print("⚠️  System partially ready")
        print("\nModel exists but no training history in database.")
        print("  1. Start services: python start_services.py")
        print("  2. Make predictions: python populate_dashboard.py")
        
    else:
        print("❌ System needs setup")
        print("\nRun these commands in order:")
        print("  1. python data_generator.py      # Generate training data")
        print("  2. python train_model.py          # Train model")
        print("  3. python start_services.py       # Start all services")
        print("  4. python populate_dashboard.py   # Make predictions")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

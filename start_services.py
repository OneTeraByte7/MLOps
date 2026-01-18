"""
Start all MLOps services: API, MLflow, and React Dashboard
"""
import subprocess
import sys
import time
import os
import signal
from dotenv import load_dotenv

load_dotenv()

processes = []

def start_mlflow():
    """Start MLflow tracking server"""
    print("🚀 Starting MLflow Tracking Server...")
    # Use Supabase/Postgres backend if SUPABASE_URL is present in environment
    supabase_db = os.environ.get('SUPABASE_URL') or os.environ.get('DATABASE_URL')
    cmd = [sys.executable, "-m", "mlflow", "server"]
    env = os.environ.copy()

    if supabase_db:
        backend = supabase_db
        if backend.startswith('postgresql://'):
            backend = backend.replace('postgresql://', 'postgresql+psycopg2://', 1)

        cmd += [
            "--backend-store-uri", backend,
            "--default-artifact-root", "s3://mlflow-artifacts",
            "--host", "0.0.0.0",
            "--port", "5000"
        ]

        # Configure S3-compatible endpoint for Supabase Storage (if project id available)
        supabase_proj = os.environ.get('SUPABASE_PROJECT_ID')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_ANON_KEY') or os.environ.get('SUPABSE_ANON_KEY')
        if supabase_proj:
            env['MLFLOW_S3_ENDPOINT_URL'] = f"https://{supabase_proj}.supabase.co/storage/v1"
        if supabase_key:
            env['AWS_ACCESS_KEY_ID'] = supabase_key
            env['AWS_SECRET_ACCESS_KEY'] = supabase_key

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    else:
        cmd += [
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", "0.0.0.0",
            "--port", "5000"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    processes.append(proc)
    time.sleep(3)
    print("✓ MLflow running on http://localhost:5000")
    return proc

def start_api():
    """Start FastAPI server"""
    print("\n🚀 Starting FastAPI Server...")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.app:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    proc = subprocess.Popen(cmd)
    processes.append(proc)
    time.sleep(3)
    print("✓ API running on http://localhost:8000")
    print("✓ API Docs available at http://localhost:8000/docs")
    return proc

def start_react():
    """Start React development server"""
    print("\n🚀 Starting React Dashboard...")
    
    frontend_dir = "dashboard/react-frontend"
    
    if not os.path.exists(frontend_dir):
        print(f"✗ React frontend directory not found: {frontend_dir}")
        return None
    
    # Check if npm is available
    try:
        npm_check = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            shell=True
        )
        if npm_check.returncode != 0:
            print("✗ npm not found. Please install Node.js from https://nodejs.org/")
            print("  Skipping React dashboard...")
            return None
    except Exception as e:
        print(f"✗ npm not found: {e}")
        print("  Install Node.js from https://nodejs.org/")
        print("  Skipping React dashboard...")
        return None
    
    # Check if node_modules exists
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("📦 Installing React dependencies (this may take a minute)...")
        npm_install = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            shell=True
        )
        if npm_install.returncode != 0:
            print("✗ npm install failed")
            return None
    
    cmd = ["npm", "run", "dev"]
    proc = subprocess.Popen(cmd, cwd=frontend_dir, shell=True)
    processes.append(proc)
    time.sleep(3)
    print("✓ React Dashboard running on http://localhost:5173")
    return proc

def cleanup(signum=None, frame=None):
    """Stop all services"""
    print("\n\n🛑 Shutting down all services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print("✓ All services stopped")
    sys.exit(0)

def main():
    print("=" * 70)
    print("MLOps Churn Prediction - Full Stack Startup")
    print("=" * 70)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Start services
        mlflow_proc = start_mlflow()
        api_proc = start_api()
        react_proc = start_react()
        
        print("\n" + "=" * 70)
        if react_proc:
            print("✓ All services running!")
        else:
            print("✓ Backend services running! (React skipped)")
        print("=" * 70)
        print("\n📊 Service URLs:")
        if react_proc:
            print("  • React Dashboard:  http://localhost:5173")
        else:
            print("  • React Dashboard:  SKIPPED (npm not found)")
        print("  • FastAPI Backend:  http://localhost:8000")
        print("  • API Docs:         http://localhost:8000/docs")
        print("  • MLflow UI:        http://localhost:5000")
        print("\n💡 Tips:")
        print("  • Dashboard reads REAL prediction data from Supabase (if configured)")
        print("  • To make predictions: python populate_dashboard.py")
        if not react_proc:
            print("  • To start React manually: cd dashboard/react-frontend && npm run dev")
        print("  • To stop all services: Press Ctrl+C")
        print("=" * 70)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Check if any process died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠ A service stopped unexpectedly (exit code: {proc.returncode})")
                    cleanup()
                    
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        cleanup()

if __name__ == "__main__":
    main()

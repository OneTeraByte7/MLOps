import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_PROJECT_ID = os.environ.get('SUPABASE_PROJECT_ID')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_ANON_KEY')

def load_csv_to_supabase(csv_path: str, table: str = 'customers'):
    """Load customer CSV into Supabase `customers` table. CSV columns should match schema.
    This replaces synthetic generation; it uploads real customer data to Supabase."""
    if not (SUPABASE_PROJECT_ID and SUPABASE_KEY):
        raise RuntimeError('Supabase credentials not found in environment (.env)')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    try:
        from supabase import create_client
        sb_url = f"https://{SUPABASE_PROJECT_ID}.supabase.co"
        sb = create_client(sb_url, SUPABASE_KEY)
    except Exception as e:
        raise RuntimeError(f'Could not initialize Supabase client: {e}')

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient='records')

    # Insert in batches
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        res = sb.table(table).insert(batch).execute()
        # No explicit handling here; Supabase response can be inspected if needed

    print(f'✓ Uploaded {len(records)} records from {csv_path} to Supabase table `{table}`')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python data_generator.py path/to/customers.csv')
        sys.exit(1)
    csv_file = sys.argv[1]
    load_csv_to_supabase(csv_file)

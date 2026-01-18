"""Prediction Logger - send predictions to Supabase. Falls back to in-memory cache if Supabase not configured.
This logger does NOT persist JSON files on disk.
"""
from datetime import datetime
from typing import Dict, List
import threading
from dotenv import load_dotenv
import os

load_dotenv()

# Supabase settings
SUPABASE_PROJECT_ID = os.environ.get('SUPABASE_PROJECT_ID') or os.environ.get('SUPABASE_PROJECT')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_ANON_KEY') or os.environ.get('SUPABSE_ANON_KEY')
SUPABASE_URL = os.environ.get('SUPABASE_URL')

class PredictionLogger:
    def __init__(self, max_predictions: int = 10000):
        self.max_predictions = max_predictions
        self.predictions: List[Dict] = []  # in-memory cache only
        self.lock = threading.Lock()

        # Try to initialize official Supabase client (preferred)
        self.supabase_client = None
        try:
            from supabase import create_client
            if SUPABASE_URL and SUPABASE_KEY:
                self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception:
            self.supabase_client = None

        # Fallback: raw REST endpoint if project id and key available
        if not self.supabase_client and SUPABASE_PROJECT_ID and SUPABASE_KEY:
            self.supabase_url = f"https://{SUPABASE_PROJECT_ID}.supabase.co/rest/v1/predictions"
            self.supabase_headers = {
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            }
        else:
            self.supabase_url = None
            self.supabase_headers = None

    def _load_predictions(self):
        # no-op: we do not persist predictions to disk
        return

    def log_prediction(self, prediction_data: Dict):
        """Log a single prediction. Try Supabase insert, fallback to local file storage."""
        with self.lock:
            # Add timestamp if missing
            if 'timestamp' not in prediction_data:
                prediction_data['timestamp'] = datetime.now().isoformat()
            # Attempt Supabase insert (preferred: official client)
            if self.supabase_client:
                try:
                    res = self.supabase_client.table('predictions').insert(prediction_data).execute()
                    if res and (getattr(res, 'status_code', 200) in (200, 201) or getattr(res, 'data', None)):
                        # keep in-memory cache for quick access
                        self.predictions.append(prediction_data)
                        if len(self.predictions) > self.max_predictions:
                            self.predictions = self.predictions[-self.max_predictions:]
                        return
                    else:
                        print(f"⚠ Supabase client insert returned: {res}")
                except Exception as e:
                    print(f"⚠ Supabase client insert error: {e}")

            # Fallback to REST
            if self.supabase_url:
                try:
                    import requests
                    # Retry a few times for transient network issues (Supabase can be flaky)
                    attempts = 3
                    for a in range(attempts):
                        try:
                            resp = requests.post(self.supabase_url, headers=self.supabase_headers, json=[prediction_data], timeout=15)
                            if resp.status_code in (200, 201):
                                self.predictions.append(prediction_data)
                                if len(self.predictions) > self.max_predictions:
                                    self.predictions = self.predictions[-self.max_predictions:]
                                return
                            else:
                                # non-2xx response - log and break (no point retrying for auth errors)
                                print(f"⚠ Supabase REST insert failed: {resp.status_code} {resp.text}")
                                break
                        except requests.exceptions.RequestException as re:
                            if a < attempts - 1:
                                # backoff
                                import time
                                time.sleep(1 + a)
                                continue
                            print(f"⚠ Supabase REST insert error after {attempts} attempts: {re}")
                            break
                except Exception as e:
                    print(f"⚠ Supabase REST insert error: {e}")

            # If no Supabase available, keep in-memory only (do NOT write to disk)
            self.predictions.append(prediction_data)
            if len(self.predictions) > self.max_predictions:
                self.predictions = self.predictions[-self.max_predictions:]

    def _save_predictions(self):
        # intentionally disabled: do not persist predictions to disk
        return

    def get_predictions(self, limit: int = None) -> List[Dict]:
        """Get recent predictions (from local cache). For full history, query Supabase directly."""
        # If Supabase client is configured, attempt to fetch recent predictions from Supabase
        if self.supabase_client:
            try:
                q = self.supabase_client.table('predictions').select('*')
                # order by timestamp desc
                q = q.order('timestamp', desc=True)
                if limit:
                    q = q.limit(limit)
                res = q.execute()
                data = res.data if hasattr(res, 'data') else res
                # Ensure data is list and return in chronological order (oldest first)
                if isinstance(data, list):
                    return list(reversed(data)) if not limit else list(reversed(data))
            except Exception as e:
                print(f"⚠ Supabase predictions fetch error: {e}")

        # If we have a REST endpoint configured (fallback), try fetching via REST
        if not self.supabase_client and self.supabase_url and self.supabase_headers:
            try:
                import requests
                params = {'select': '*', 'order': 'timestamp.desc'}
                if limit:
                    params['limit'] = limit
                attempts = 3
                for a in range(attempts):
                    try:
                        resp = requests.get(self.supabase_url, headers=self.supabase_headers, params=params, timeout=15)
                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list):
                                # Supabase returns newest-first; return oldest-first for the dashboard
                                return list(reversed(data)) if data else []
                            return []
                        else:
                            print(f"⚠ Supabase REST fetch failed: {resp.status_code} {resp.text}")
                            break
                    except requests.exceptions.RequestException as re:
                        if a < attempts - 1:
                            import time
                            time.sleep(1 + a)
                            continue
                        print(f"⚠ Supabase REST fetch error after {attempts} attempts: {re}")
                        break
            except Exception as e:
                print(f"⚠ Supabase REST fetch error: {e}")

        # Fallback to local cache
        with self.lock:
            if limit:
                return self.predictions[-limit:]
            return self.predictions.copy()

    def flush(self):
        """Force save to disk"""
        # no-op: we do not persist to disk
        return


# Global instance
prediction_logger = PredictionLogger()

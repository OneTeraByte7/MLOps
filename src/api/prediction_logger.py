"""
Prediction Logger - Store predictions for dashboard
"""
import json
import os
from datetime import datetime
from typing import Dict, List
import threading

class PredictionLogger:
    def __init__(self, log_file='data/recent_predictions.json', max_predictions=10000):
        self.log_file = log_file
        self.max_predictions = max_predictions
        self.predictions = []
        self.lock = threading.Lock()
        self._load_predictions()
    
    def _load_predictions(self):
        """Load existing predictions from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.predictions = json.load(f)
                print(f"✓ Loaded {len(self.predictions)} predictions from {self.log_file}")
            except Exception as e:
                print(f"⚠ Could not load predictions: {e}")
                self.predictions = []
    
    def log_prediction(self, prediction_data: Dict):
        """Log a single prediction"""
        with self.lock:
            self.predictions.append(prediction_data)
            
            # Keep only recent predictions
            if len(self.predictions) > self.max_predictions:
                self.predictions = self.predictions[-self.max_predictions:]
            
            # Save to file periodically (every 10 predictions)
            if len(self.predictions) % 10 == 0:
                self._save_predictions()
    
    def _save_predictions(self):
        """Save predictions to file"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self.predictions, f)
        except Exception as e:
            print(f"⚠ Could not save predictions: {e}")
    
    def get_predictions(self, limit: int = None) -> List[Dict]:
        """Get recent predictions"""
        with self.lock:
            if limit:
                return self.predictions[-limit:]
            return self.predictions.copy()
    
    def flush(self):
        """Force save to disk"""
        with self.lock:
            self._save_predictions()

# Global instance
prediction_logger = PredictionLogger()

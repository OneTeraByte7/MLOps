import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generator_churn_data(n_customers=10000, output_dir='data/raw')
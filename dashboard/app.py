import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import mlflow
import sys
sys.path.append('src')

st.set_page_config(
    page_title = "Churn Prediction MLOps Dashboard",
    page_icon=":bar_chart:",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_mlflow_experiemnts():
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("churn-prediction")
        
        runs = mlflow.search_runs(order_by = ["start_time DESC"], max_results = 20)
        return runs
    
    except:
        return pd.DataFrame()
    
@st.cache_data(ttl = 300)
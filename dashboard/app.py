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
def load_mlflow_experiments():
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("churn-prediction")
        
        runs = mlflow.search_runs(order_by = ["start_time DESC"], max_results = 20)
        return runs
    
    except:
        return pd.DataFrame()
    
@st.cache_data(ttl = 300)
def load_drift_report():
    try:
        with open('monitoring/reports/latest_report.json', 'r') as f:
            return json.load(f)
        
    except:
        return None
    
@st.cache_data(ttl = 60)
def load_recent_predictions():
    np.random.seed(42)
    dates = pd.date_range(end = datetime.now(), periods = 1000, freq = '1H')
    
    data = {
        'timestamp': dates,
        'customer_id': [f'CUST_{i:06d}' for i in np.ranodom.randit(0, 10000, 1000)],
        'churn_probability': np.random.beta(2, 5, 1000),
        'prediction': np.random.choice(['Yes', 'No'], 1000, p = [0.25, 0.75]),
        'risk_level': np.random.choice(['High', 'Mdedium', 'Low'], 1000, p = [0.15, 0.35, 0.50]),
        'model_version': np.random.choice(['v1.0.0', 'v1.1.0'], 1000, p = [0.6, 0.4])
    }
    
    return pd.DataFrame(data)

st.sidebar.markdown("## MLOps Dashboard")
st.sidebar.markdown("###Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Model Performnace", "Drift Analysis", 'Predictions', "Explainability"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("###Settings")
auto_refresh = st.sidebar.checkbox("Auto Refresh", values = False)

if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
    
st.sidebar.markdown("---")
st.sidebar.markdown("###System Staus")
st.sidebar.markdown("@API: Online")
st.sidebar.markdown("@MLflow: COnnected")
st.sidebar.markdown(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")


if page == "Overview":
    st.markdown('<div class="main-header">🎯 Churn Prediction System Overview</div>', 
                unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    predictions_df = load_recent_predictions()
    
    with col1:
        st.metric(
            label="Total Predictions (24h)",
            value=f"{len(predictions_df[predictions_df['timestamp'] > datetime.now() - timedelta(hours=24)]):,}",
            delta="+12%"
        )
    
    with col2:
        churn_rate = predictions_df[predictions_df['prediction'] == 'Yes'].shape[0] / len(predictions_df)
        st.metric(
            label="Predicted Churn Rate",
            value=f"{churn_rate:.1%}",
            delta="-2.3%"
        )
    
    with col3:
        high_risk = predictions_df[predictions_df['risk_level'] == 'High'].shape[0]
        st.metric(
            label="High Risk Customers",
            value=f"{high_risk:,}",
            delta="+5"
        )
    
    with col4:
        avg_confidence = predictions_df['churn_probability'].mean()
        st.metric(
            label="Avg Confidence",
            value=f"{avg_confidence:.2%}",
            delta="+1.2%"
        )
    
    # Prediction Volume Over Time
    st.markdown("### 📈 Prediction Volume")
    
    hourly_predictions = predictions_df.groupby(
        predictions_df['timestamp'].dt.floor('H')
    ).size().reset_index(name='count')
    
    fig_volume = px.line(
        hourly_predictions,
        x='timestamp',
        y='count',
        title='Predictions per Hour',
        labels={'count': 'Number of Predictions', 'timestamp': 'Time'}
    )
    fig_volume.update_layout(height=300)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Risk Level Distribution")
        risk_counts = predictions_df['risk_level'].value_counts()
        
        fig_risk = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=['#f44336', '#ff9800', '#4caf50']
        )])
        fig_risk.update_layout(height=300)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Churn Probability Distribution")
        
        fig_dist = px.histogram(
            predictions_df,
            x='churn_probability',
            nbins=50,
            title='Distribution of Churn Probabilities',
            labels={'churn_probability': 'Churn Probability'}
        )
        fig_dist.update_layout(height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Recent High-Risk Customers
    st.markdown("### ⚠️ Recent High-Risk Customers")
    
    high_risk_df = predictions_df[
        predictions_df['risk_level'] == 'High'
    ].sort_values('timestamp', ascending=False).head(10)
    
    display_df = high_risk_df[['timestamp', 'customer_id', 'churn_probability', 'model_version']].copy()
    display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x:.1%}")
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    

elif page == "Model Performance":
    st.markdown('<div class="main-header">🎯 Model Performance Tracking</div>', 
                unsafe_allow_html=True)
    
    # Load MLflow data
    runs_df = load_mlflow_experiments()
    
    if not runs_df.empty:
        # Performance Metrics Over Time
        st.markdown("### 📊 Performance Metrics Over Time")
        
        metrics_to_plot = ['metrics.test_auc', 'metrics.test_f1', 'metrics.test_precision', 'metrics.test_recall']
        available_metrics = [m for m in metrics_to_plot if m in runs_df.columns]
        
        if available_metrics:
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=['AUC', 'F1 Score', 'Precision', 'Recall']
            )
            
            metric_names = ['test_auc', 'test_f1', 'test_precision', 'test_recall']
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for metric, pos, name in zip(available_metrics, positions, metric_names):
                fig_metrics.add_trace(
                    go.Scatter(
                        x=runs_df['start_time'],
                        y=runs_df[metric],
                        mode='lines+markers',
                        name=name.replace('test_', '').upper()
                    ),
                    row=pos[0], col=pos[1]
                )
            
            fig_metrics.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Latest Model Metrics
        st.markdown("### 🏆 Latest Model Performance")
        
        if len(runs_df) > 0:
            latest_run = runs_df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "AUC",
                    f"{latest_run.get('metrics.test_auc', 0):.4f}"
                )
            
            with col2:
                st.metric(
                    "F1 Score",
                    f"{latest_run.get('metrics.test_f1', 0):.4f}"
                )
            
            with col3:
                st.metric(
                    "Precision",
                    f"{latest_run.get('metrics.test_precision', 0):.4f}"
                )
            
            with col4:
                st.metric(
                    "Recall",
                    f"{latest_run.get('metrics.test_recall', 0):.4f}"
                )
        
        # Model Comparison
        st.markdown("### 🔄 Model Comparison")
        
        comparison_df = runs_df[[
            'start_time', 
            'metrics.test_auc', 
            'metrics.test_f1',
            'params.max_depth',
            'params.learning_rate'
        ]].copy()
        
        comparison_df.columns = ['Date', 'AUC', 'F1', 'Max Depth', 'Learning Rate']
        comparison_df['Date'] = pd.to_datetime(comparison_df['Date']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(comparison_df.head(10), use_container_width=True, hide_index=True)
    
    else:
        st.warning("No MLflow experiments found. Train a model first!")
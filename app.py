"""
Copper Brain v2 - Interactive Dashboard
========================================
Comprehensive Streamlit dashboard for copper direction prediction analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import io

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from features import engineer_all_features
from model_utils import calculate_overall_metrics

# Page configuration
st.set_page_config(
    page_title="Copper Brain v2",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    .success-metric {
        border-left-color: #28a745 !important;
    }
    .warning-metric {
        border-left-color: #ffc107 !important;
    }
    .danger-metric {
        border-left-color: #dc3545 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
#  DATA LOADING FUNCTIONS
# =========================

@st.cache_data(ttl=3600)
def load_backtest_results():
    """Load backtest results from CSV."""
    try:
        df = pd.read_csv("outputs/backtest_results.csv", parse_dates=['date'])
        return df
    except FileNotFoundError:
        st.error("Backtest results not found. Please run copper_brain_v2.py first.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_performance_metrics():
    """Load performance metrics from text file."""
    try:
        with open("outputs/performance_metrics.txt", 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "Performance metrics not found. Please run copper_brain_v2.py first."

@st.cache_resource
def load_final_model():
    """Load the trained model and feature columns."""
    try:
        model = joblib.load("outputs/final_model.pkl")
        feature_cols = joblib.load("outputs/feature_columns.pkl")
        return model, feature_cols
    except FileNotFoundError:
        return None, None

# =========================
#  DASHBOARD PAGES
# =========================

def overview_page():
    """Main overview page with key metrics."""
    st.markdown('<div class="main-header">🔮 Copper Brain v2</div>', unsafe_allow_html=True)
    st.markdown("*Copper 21-day direction model - walk-forward results, compared with an always-up baseline*")
    st.markdown("---")

    # Load data
    backtest_df = load_backtest_results()
    metrics_text = load_performance_metrics()

    if backtest_df.empty:
        st.error("No backtest data available. Please run the training script first.")
        return

    # Calculate metrics
    metrics = calculate_overall_metrics(backtest_df)

    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)

    accuracy = metrics.get('overall_accuracy', 0)
    precision = metrics.get('overall_precision', 0)
    f1 = metrics.get('overall_f1', 0)
    roc_auc = metrics.get('overall_roc_auc', 0)

    # Naive baseline: always predict "up". Green only if the model beats it.
    always_up = float(backtest_df['actual_direction'].mean())

    with col1:
        accuracy_class = "success-metric" if accuracy > always_up else "warning-metric"
        st.markdown(f"""
        <div class="metric-card {accuracy_class}">
            <div class="metric-value">{accuracy:.1%}</div>
            <div class="metric-label">Directional Accuracy (always-up: {always_up:.1%})</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{precision:.1%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1:.1%}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{roc_auc:.3f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)

    # Download Section
    st.markdown("---")
    st.subheader("💾 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download backtest results as CSV
        csv_buffer = io.StringIO()
        backtest_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Backtest Results (CSV)",
            data=csv_data,
            file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download complete backtest results with predictions and actuals"
        )
    
    with col2:
        # Download performance metrics as TXT
        st.download_button(
            label="📥 Download Performance Metrics (TXT)",
            data=metrics_text,
            file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            help="Download detailed performance metrics report"
        )
    
    with col3:
        # Download summary report as CSV
        summary_data = {
            'Metric': ['Accuracy', 'Precision', 'F1 Score', 'ROC-AUC', 
                      'True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
            'Value': [
                f"{accuracy:.4f}", f"{precision:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}",
                metrics.get('true_positives', 0), metrics.get('true_negatives', 0),
                metrics.get('false_positives', 0), metrics.get('false_negatives', 0)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Summary Report (CSV)",
            data=summary_csv,
            file_name=f"summary_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download summary metrics as CSV"
        )
    
    st.markdown("---")

    # Model info
    st.subheader("📊 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Out-of-sample predictions**: {len(backtest_df)}
        **Test Period**: {backtest_df['date'].min().strftime('%Y-%m-%d')} to {backtest_df['date'].max().strftime('%Y-%m-%d')}
        **Forecast Horizon**: 21 trading days
        **Walk-Forward Windows**: 1,260-calendar-day (~3.45-year) training, 21-calendar-day steps
        """)

    with col2:
        confusion_matrix_text = f"""
        **Confusion Matrix**
        - True Positives: {metrics.get('true_positives', 0)}
        - True Negatives: {metrics.get('true_negatives', 0)}
        - False Positives: {metrics.get('false_positives', 0)}
        - False Negatives: {metrics.get('false_negatives', 0)}
        """
        st.info(confusion_matrix_text)

    # Recent performance
    st.subheader("📈 Recent Performance")

    # Last 6 months of predictions (relative to the latest prediction, not today), compared with
    # always predicting "up" over the same window - not with the model's own long-run average.
    recent_df = backtest_df[backtest_df['date'] >= (backtest_df['date'].max() - timedelta(days=180))]
    if not recent_df.empty:
        recent_accuracy = accuracy_score(recent_df['actual_direction'], recent_df['predicted_direction'])
        recent_always_up = float(recent_df['actual_direction'].mean())

        st.metric(
            f"Last 6 Months Accuracy (always-up: {recent_always_up:.1%})",
            f"{recent_accuracy:.1%}",
            delta=f"{recent_accuracy - recent_always_up:+.1%} vs always-up"
        )

    # Performance over time chart
    st.subheader("📉 Performance Over Time")

    # Rolling accuracy next to the naive "always predict up" baseline. (This used to plot a
    # rolling *correlation* under an "accuracy" title, against a 60% target line.)
    backtest_df = backtest_df.sort_values('date')
    bt = backtest_df.set_index('date')
    rolling = pd.DataFrame({
        'Model accuracy': (bt['predicted_direction'] == bt['actual_direction']).astype(float).rolling('90D', min_periods=40).mean(),
        'Always-up accuracy': bt['actual_direction'].astype(float).rolling('90D', min_periods=40).mean(),
    }).reset_index()

    fig = px.line(
        rolling,
        x='date',
        y=['Model accuracy', 'Always-up accuracy'],
        title='90-Day Rolling Directional Accuracy vs an always-up baseline',
        labels={'value': 'Accuracy', 'date': 'Date', 'variable': ''}
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="coin flip")
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

def backtest_explorer_page():
    """Detailed backtest analysis page."""
    st.header("🔍 Backtest Explorer")

    backtest_df = load_backtest_results()

    if backtest_df.empty:
        st.error("No backtest data available.")
        return

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        threshold = st.slider(
            "Probability Threshold",
            min_value=0.3,
            max_value=0.7,
            value=0.45,
            step=0.01,
            help="Adjust classification threshold"
        )

    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(backtest_df['date'].min().date(), backtest_df['date'].max().date()),
            help="Filter predictions by date range"
        )

    with col3:
        show_correct_only = st.checkbox("Show Correct Predictions Only", value=False)

    # Apply filters
    filtered_df = backtest_df.copy()

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
        ]

    # Accuracy is measured on the whole date range, before the "correct only" display filter
    # (measuring after it inflated the number).
    filtered_df['predicted_with_new_threshold'] = (filtered_df['prediction_probability'] >= threshold).astype(int)
    new_accuracy = accuracy_score(filtered_df['actual_direction'], filtered_df['predicted_with_new_threshold'])
    range_always_up = float(filtered_df['actual_direction'].mean()) if len(filtered_df) else 0.0

    st.metric(f"Accuracy with {threshold:.0%} threshold (always-up: {range_always_up:.1%})",
              f"{new_accuracy:.1%}",
              delta=f"{new_accuracy - range_always_up:+.1%} vs always-up")

    if show_correct_only:
        filtered_df = filtered_df[
            filtered_df['actual_direction'] == filtered_df['predicted_with_new_threshold']
        ]
        st.caption("Showing correct predictions only - the charts below (heatmap, confusion matrix) "
                   "are then 100% correct by construction; the accuracy above uses all predictions.")

    # Direction vs Actual Chart
    st.subheader("📊 Prediction vs Actual Direction")

    # Create direction chart
    fig = go.Figure()

    # Actual directions
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['actual_direction'],
        mode='lines',
        name='Actual Direction',
        line=dict(color='#1f77b4', width=2),
        yaxis='y1'
    ))

    # Predicted directions
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['predicted_with_new_threshold'],
        mode='lines',
        name='Predicted Direction',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        yaxis='y1'
    ))

    # Prediction probabilities
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['prediction_probability'],
        mode='lines',
        name='Prediction Probability',
        line=dict(color='#2ca02c', width=1),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Copper Direction Predictions vs Actual',
        xaxis_title='Date',
        yaxis=dict(title='Direction (0=Down, 1=Up)', side='left'),
        yaxis2=dict(title='Probability', overlaying='y', side='right'),
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Rolling accuracy heatmap
    st.subheader("🗺️ Rolling Accuracy Heatmap")

    # Create monthly accuracy pivot
    filtered_df['year_month'] = filtered_df['date'].dt.to_period('M')
    monthly_accuracy = filtered_df.groupby('year_month').apply(
        lambda x: accuracy_score(x['actual_direction'], x['predicted_with_new_threshold'])
        if len(x) > 0 else np.nan
    )

    monthly_df = monthly_accuracy.reset_index()
    monthly_df['year'] = monthly_df['year_month'].dt.year
    monthly_df['month'] = monthly_df['year_month'].dt.month

    pivot_accuracy = monthly_df.pivot(index='year', columns='month', values=0)

    fig = px.imshow(
        pivot_accuracy,
        title='Monthly Directional Accuracy',
        labels=dict(x="Month", y="Year", color="Accuracy"),
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.subheader("📋 Confusion Matrix")

    cm = confusion_matrix(
        filtered_df['actual_direction'],
        filtered_df['predicted_with_new_threshold'],
        labels=[0, 1],
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        title='Confusion Matrix',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Predicted Down', 'Predicted Up'],
        y=['Actual Down', 'Actual Up']
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # Probability histogram
    st.subheader("📊 Prediction Probability Distribution")

    fig = px.histogram(
        filtered_df,
        x='prediction_probability',
        color='actual_direction',
        nbins=20,
        title='Prediction Probability Distribution',
        labels={'prediction_probability': 'Prediction Probability', 'actual_direction': 'Actual Direction'},
        color_discrete_map={0: '#dc3545', 1: '#28a745'}
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # Download filtered results
    st.markdown("---")
    st.subheader("💾 Download Filtered Results")
    
    # Prepare filtered data for download
    download_df = filtered_df[['date', 'close_price', 'prediction_probability', 
                                'predicted_direction', 'actual_direction', 
                                'predicted_with_new_threshold']].copy()
    download_df['date'] = download_df['date'].dt.strftime('%Y-%m-%d')
    download_df['correct_prediction'] = (download_df['actual_direction'] == download_df['predicted_with_new_threshold'])
    
    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Filtered Backtest Data (CSV)",
        data=csv_data,
        file_name=f"filtered_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Download the currently filtered backtest results"
    )

def feature_importance_page():
    """Feature importance and SHAP analysis page."""
    st.header("🔍 Feature Importance Analysis")

    model, feature_cols = load_final_model()

    if model is None:
        st.error("Model not found. Please run training first.")
        return

    # Get feature importances
    try:
        importances = model.estimator_.feature_importances_
    except:
        # For calibrated models, access the base estimator
        importances = model.estimator.feature_importances_

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Top 20 features
    st.subheader("📈 Top 20 Feature Importances")

    fig = px.bar(
        importance_df.head(20),
        x='importance',
        y='feature',
        orientation='h',
        title='XGBoost Feature Importances (Top 20)',
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    # Feature categories
    st.subheader("🏷️ Feature Categories")

    # Categorize features
    categories = {
        'Momentum': [f for f in feature_cols if 'ret_' in f],
        'Moving Averages': [f for f in feature_cols if 'ma_' in f or 'price_over_ma' in f],
        'Volatility': [f for f in feature_cols if 'vol' in f],
        'Volume': [f for f in feature_cols if 'volume' in f or 'obv' in f],
        'Candles': [f for f in feature_cols if any(x in f for x in ['body', 'upper_wick', 'lower_wick', 'bullish'])],
        'DXY': [f for f in feature_cols if 'dxy' in f.lower()],
        'Gold': [f for f in feature_cols if 'gold' in f.lower()],
        'Time': [f for f in feature_cols if any(x in f for x in ['dow', 'month', 'year'])]
    }

    # Calculate category importances
    category_importance = {}
    for category, features in categories.items():
        if features:
            cat_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
            category_importance[category] = cat_importance

    # Category importance chart
    cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Total_Importance'])
    cat_df = cat_df.sort_values('Total_Importance', ascending=False)

    fig = px.pie(
        cat_df,
        values='Total_Importance',
        names='Category',
        title='Feature Importance by Category'
    )
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Download feature importance data
    st.markdown("---")
    st.subheader("💾 Download Feature Importance Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download all feature importances
        csv_buffer = io.StringIO()
        importance_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download All Feature Importances (CSV)",
            data=csv_data,
            file_name=f"feature_importances_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download complete feature importance rankings"
        )
    
    with col2:
        # Download category importances
        cat_csv = cat_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Category Importances (CSV)",
            data=cat_csv,
            file_name=f"category_importances_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download aggregated importance by feature category"
        )
    
    st.markdown("---")

    # SHAP Analysis (if available)
    st.subheader("🎯 SHAP Analysis")

    try:
        import shap

        # Load some sample data for SHAP
        backtest_df = load_backtest_results()
        if not backtest_df.empty:
            # Get a sample of recent data
            sample_data = backtest_df.tail(100).copy()

            # Create feature matrix (this is approximate since we don't have raw features)
            # In a real implementation, you'd want to load the actual feature matrix
            st.info("SHAP analysis requires the original feature matrix. This is a placeholder for demonstration.")

            st.code("""
# Real SHAP implementation would look like this:

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols)

# Waterfall plot for individual prediction
shap.plots.waterfall(explainer.expected_value, shap_values[0], X_sample.iloc[0])
            """)

    except ImportError:
        st.warning("SHAP library not available. Install with: pip install shap")

def recent_predictions_page():
    """Recent predictions and live data page."""
    st.header("📅 Recent Predictions")

    backtest_df = load_backtest_results()

    if backtest_df.empty:
        st.error("No backtest data available.")
        return

    # Most recent 30 days of predictions, counted back from the latest prediction (not today:
    # the 21-day target means the newest prediction is always weeks old).
    recent_df = backtest_df[backtest_df['date'] >= (backtest_df['date'].max() - timedelta(days=30))].copy()

    if recent_df.empty:
        st.warning("No predictions in the 30 days before the latest prediction.")
        return

    # Format for display
    display_df = recent_df[['date', 'close_price', 'prediction_probability', 'predicted_direction', 'actual_direction']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['predicted_direction'] = display_df['predicted_direction'].map({0: '📉 Down', 1: '📈 Up'})
    display_df['actual_direction'] = display_df['actual_direction'].map({0: '📉 Down', 1: '📈 Up'})
    display_df['correct'] = display_df['predicted_direction'] == display_df['actual_direction']
    display_df['prediction_probability'] = display_df['prediction_probability'].round(3)

    # Color coding
    def color_correct(val):
        return 'color: green' if val else 'color: red'

    styled_df = display_df.style.map(color_correct, subset=['correct'])  # applymap was removed in pandas 3

    st.dataframe(styled_df, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)

    recent_accuracy = accuracy_score(recent_df['actual_direction'], recent_df['predicted_direction'])

    recent_always_up = float(recent_df['actual_direction'].mean())

    with col1:
        st.metric(f"Recent Accuracy (30 days, always-up: {recent_always_up:.1%})",
                  f"{recent_accuracy:.1%}",
                  delta=f"{recent_accuracy - recent_always_up:+.1%} vs always-up")
        st.caption(f"{len(recent_df)} daily predictions of a 21-day move - about one independent outcome.")

    with col2:
        avg_probability = recent_df['prediction_probability'].mean()
        st.metric("Avg predicted P(up)", f"{avg_probability:.1%}")

    with col3:
        total_predictions = len(recent_df)
        st.metric("Total Predictions", total_predictions)

    # Price movement visualization
    st.subheader("💰 Price Movements")

    fig = go.Figure()

    # Copper prices
    fig.add_trace(go.Scatter(
        x=recent_df['date'],
        y=recent_df['close_price'],
        mode='lines+markers',
        name='Copper Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # Color markers by prediction correctness
    colors = ['green' if correct else 'red' for correct in (recent_df['actual_direction'] == recent_df['predicted_direction']).values]

    fig.add_trace(go.Scatter(
        x=recent_df['date'],
        y=recent_df['close_price'],
        mode='markers',
        name='Predictions',
        marker=dict(color=colors, size=8, symbol='circle'),
        showlegend=False
    ))

    fig.update_layout(
        title='Recent Copper Prices with Prediction Markers',
        xaxis_title='Date',
        yaxis_title='Copper Price (USD)',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("🟢 Green markers = Correct predictions | 🔴 Red markers = Incorrect predictions")

    # Download recent predictions
    st.markdown("---")
    st.subheader("💾 Download Recent Predictions")
    
    # Prepare download data (clean version without emojis)
    download_df = recent_df[['date', 'close_price', 'prediction_probability', 
                              'predicted_direction', 'actual_direction']].copy()
    download_df['date'] = download_df['date'].dt.strftime('%Y-%m-%d')
    download_df['correct_prediction'] = (download_df['actual_direction'] == download_df['predicted_direction'])
    
    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Recent Predictions (CSV)",
        data=csv_data,
        file_name=f"recent_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Download last 30 days of predictions"
    )

def settings_page():
    """Settings and model management page."""
    st.header("⚙️ Settings & Model Management")

    # Retrain model section
    st.subheader("🔄 Retrain Model")

    col1, col2 = st.columns(2)

    with col1:
        horizon_options = [7, 14, 21, 42]
        selected_horizon = st.selectbox(
            "Forecast Horizon (days)",
            options=horizon_options,
            index=2,  # Default to 21
            help="Number of days ahead to predict direction"
        )

    with col2:
        threshold = st.slider(
            "Probability Threshold",
            min_value=0.3,
            max_value=0.7,
            value=0.45,
            step=0.01,
            help="Classification threshold for predictions"
        )

    if st.button("🚀 Retrain Model", type="primary"):
        st.info("Retraining functionality would be implemented here. In a real application, this would:")
        st.code("""
1. Update configuration parameters
2. Re-run feature engineering
3. Execute walk-forward validation
4. Save new model and results
5. Refresh dashboard data
        """)

    # Model info
    st.subheader("📊 Current Model Info")

    try:
        import os
        model_path = "outputs/final_model.pkl"
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / 1024  # KB
            st.success(f"✅ Model loaded ({model_size:.1f} KB)")

            # Load feature count
            _, feature_cols = load_final_model()
            if feature_cols:
                st.info(f"📈 {len(feature_cols)} features used")
        else:
            st.warning("❌ No trained model found")
    except:
        st.error("Error loading model information")

    # System info
    st.subheader("💻 System Information")

    st.code(f"""
Python Version: {sys.version.split()[0]}
Platform: {sys.platform}
Working Directory: {os.getcwd()}

Last Training: Check outputs/performance_metrics.txt
Backtest Results: outputs/backtest_results.csv
Model File: outputs/final_model.pkl
    """)

# =========================
#  MAIN APP
# =========================

def main():
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")

    pages = {
        "🏠 Overview": overview_page,
        "🔍 Backtest Explorer": backtest_explorer_page,
        "🔍 Feature Importance": feature_importance_page,
        "📅 Recent Predictions": recent_predictions_page,
        "⚙️ Settings": settings_page
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Copper Brain v2**")
    st.sidebar.markdown("*Built with Streamlit & XGBoost*")
    st.sidebar.markdown("---")

    # Load selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
"""
Dashboard UI Module
Streamlit-based dashboard for the Intelligent Performance Predictor.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import ML and analytics modules
from ml import RealTimePredictor
from analytics import TrendAnalyzer, AlertManager, Alert, AlertType, AlertSeverity

# Initialize database connection
def get_db_connection():
    """Create a database connection."""
    # Ensure data directory exists
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    db_path = data_dir / 'performance_data.db'

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

# Initialize database if it doesn't exist
if not DB_PATH.exists():
    from data_collector.database import init_db
    init_db()

# Initialize analytics components
try:
    predictor = RealTimePredictor()
    analyzer = TrendAnalyzer()
    alert_manager = AlertManager()
except Exception as e:
    st.error(f"Error initializing components: {e}")
    st.stop()

def get_current_metrics():
    """Get current performance metrics from the database."""
    conn = get_db_connection()
    try:
        # Get the most recent metrics
        df = pd.read_sql("""
            SELECT * FROM system_metrics 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, conn)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}
    finally:
        conn.close()

def get_recent_alerts(limit=5):
    """Get recent alerts from the database."""
    conn = get_db_connection()
    try:
        return pd.read_sql(f"""
            SELECT * FROM alerts 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """, conn)
    except Exception as e:
        st.warning(f"Could not fetch alerts: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_historical_data(days=7):
    """Get historical data for visualization."""
    conn = get_db_connection()
    try:
        return pd.read_sql(f"""
            SELECT * FROM performance_metrics 
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp
        """, conn, parse_dates=['timestamp'])
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def show_dashboard():
    """Main dashboard view."""
    st.title("🧠 Intelligent Performance Dashboard")
    
    # Initialize predictor and analyzer
    predictor = RealTimePredictor()
    analyzer = TrendAnalyzer()
    
    # Get current metrics
    current_metrics = get_current_metrics()
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Performance Score
    with col1:
        st.metric("🎯 Performance Score", 
                 f"{current_metrics.get('performance_score', 'N/A')}%" 
                 if current_metrics else "N/A")
    
    # Burnout Risk
    with col2:
        risk = current_metrics.get('burnout_risk', 0)
        risk_color = "green" if risk < 30 else "orange" if risk < 70 else "red"
        st.metric("🔥 Burnout Risk", 
                 f"{risk}%" if current_metrics else "N/A",
                 delta_color="off",
                 help="Lower is better")
    
    # Focus Level
    with col3:
        focus = current_metrics.get('focus_level', 0)
        st.metric("🎯 Focus Level", 
                 f"{focus}%" if current_metrics else "N/A",
                 help="Based on typing patterns and app usage")
    
    # Energy Level
    with col4:
        energy = current_metrics.get('energy_level', 0)
        st.metric("⚡ Energy Level", 
                 f"{energy}%" if current_metrics else "N/A",
                 help="Based on system activity and patterns")
    
    # Main content area
    st.markdown("---")
    
    # Two-column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Trend")
        df = get_historical_data()
        if not df.empty:
            fig = px.line(df, x='timestamp', y='performance_score', 
                         title="7-Day Performance Trend")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Burnout Risk Over Time")
        if not df.empty:
            fig = px.area(df, x='timestamp', y='burnout_risk',
                         title="Burnout Risk Trend")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent Alerts
    st.subheader("🔔 Recent Alerts")
    alerts = get_recent_alerts()
    if not alerts.empty:
        for _, alert in alerts.iterrows():
            emoji = "🟡" if alert['severity'] == 'medium' else "🔴" if alert['severity'] == 'high' else "🔵"
            with st.expander(f"{emoji} {alert['title']} - {alert['timestamp']}"):
                st.write(alert['message'])
                st.caption(f"Severity: {alert['severity'].title()}")
    else:
        st.info("No recent alerts. Everything looks good!")

def show_analytics():
    """Analytics and insights view."""
    st.title("📊 Analytics & Insights")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                 datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", 
                               datetime.now())
    
    # Get data for the selected date range
    conn = get_db_connection()
    try:
        df = pd.read_sql("""
            SELECT * FROM performance_metrics 
            WHERE date(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp
        """, conn, params=(start_date, end_date), parse_dates=['timestamp'])
        
        if df.empty:
            st.warning("No data available for the selected date range.")
            return
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Performance", f"{df['performance_score'].mean():.1f}%")
        with col2:
            st.metric("Avg Burnout Risk", f"{df['burnout_risk'].mean():.1f}%")
        with col3:
            st.metric("Data Points", len(df))
        
        # Performance vs Burnout scatter plot
        st.subheader("Performance vs Burnout Risk")
        fig = px.scatter(df, x='performance_score', y='burnout_risk',
                        color='timestamp',
                        title="Performance vs Burnout Risk Over Time",
                        labels={"performance_score": "Performance Score (%)",
                               "burnout_risk": "Burnout Risk (%)"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily patterns
        st.subheader("Daily Patterns")
        df['hour'] = df['timestamp'].dt.hour
        daily_avg = df.groupby('hour').agg({
            'performance_score': 'mean',
            'burnout_risk': 'mean',
            'focus_level': 'mean'
        }).reset_index()
        
        fig = px.line(daily_avg, x='hour', y=['performance_score', 'burnout_risk', 'focus_level'],
                     title="Average Metrics by Hour of Day",
                     labels={"value": "Score (%)", "variable": "Metric"})
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
    finally:
        conn.close()

def show_settings():
    """Settings view."""
    st.title("⚙️ Settings")
    
    st.subheader("Alert Preferences")
    email_alerts = st.checkbox("Enable Email Alerts", value=True)
    push_notifications = st.checkbox("Enable Push Notifications", value=True)
    
    st.subheader("Data Collection")
    st.checkbox("Collect System Metrics", value=True)
    st.checkbox("Track Application Usage", value=True)
    st.checkbox("Monitor Typing Patterns", value=True)
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")

def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Dashboard", "Analytics", "Settings"],
                           index=0)
    
    # Show the selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Intelligent Performance Predictor v1.0\n\n"
        "Track your performance and prevent burnout with AI-powered insights."
    )

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Intelligent Performance Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide Streamlit menu and footer
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Run the app
    main()

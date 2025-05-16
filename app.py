import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
from model import DemandPredictor
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
from database import init_db, get_session, initialize_stores, Store, DataUpload, StoreData
from data_upload import data_upload_section, get_store_data

# Set page config
st.set_page_config(
    page_title="Stockify",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define users
USERS = {
    "admin": {
        "password": "admin123",
        "role": "admin",
        "store_id": None,
        "area": None
    },
    "store1": {
        "password": "store123",
        "role": "store",
        "store_id": 1,
        "area": "Delhi"
    },
    "store2": {
        "password": "store123",
        "role": "store",
        "store_id": 2,
        "area": "Mumbai"
    },
    "store3": {
        "password": "store123",
        "role": "store",
        "store_id": 3,
        "area": "Bangalore"
    }
}

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: #f8fafc;
        padding: 1rem;
    }
    
    /* Gradient header */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Login container */
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: #ffffff;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Title container */
    .title-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 2px rgba(42, 82, 152, 0.2);
    }
    
    /* Metrics */
    .custom-metric {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .custom-metric h3 {
        color: #1e3c72;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .custom-metric p {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2a5298;
        margin: 0;
    }
    
    /* Charts */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3c72;
        margin: 0;
        padding: 0;
    }
    
    .section-header {
        margin: 2rem 0 1rem 0;
        padding: 0;
        color: #1e3c72;
        font-size: 1.5rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# User credentials
USERS = {
    'admin': {
        'password': 'admin123',
        'role': 'admin'
    },
    'store1': {
        'password': 'store1pass',
        'role': 'store',
        'store_id': 'STORE001',
        'area': 'Nagpur East'
    },
    'store2': {
        'password': 'store2pass',
        'role': 'store',
        'store_id': 'STORE002',
        'area': 'Nagpur West'
    },
    'store3': {
        'password': 'store3pass',
        'role': 'store',
        'store_id': 'STORE003',
        'area': 'Nagpur North'
    },
    'store4': {
        'password': 'store4pass',
        'role': 'store',
        'store_id': 'STORE004',
        'area': 'Nagpur South'
    },
    'store5': {
        'password': 'store5pass',
        'role': 'store',
        'store_id': 'STORE005',
        'area': 'Nagpur Central'
    }
}

def login():
    st.markdown('<div class="title-container"><h1>Stockify</h1><h3>Stock Smarter, Sell Better</h3></div>', unsafe_allow_html=True)
    

    st.subheader("Welcome Back! 👋")
    username = st.text_input("Username", key="username_input")
    password = st.text_input("Password", type="password", key="password_input")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Login", key="login_button"):
            if username in USERS and USERS[username]['password'] == password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['role'] = USERS[username]['role']
                if USERS[username]['role'] == 'store':
                    st.session_state['store_id'] = USERS[username]['store_id']
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    st.markdown('</div>', unsafe_allow_html=True)


def load_admin_data():
    try:
        df = pd.read_csv("retail_data_historical.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return None

def load_store_data():
    try:
        df = pd.read_csv("retail_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading store data: {str(e)}")
        return None

def predict_seasonal_trends(df, periods=30):
    if df.empty:
        return {}
        
    # Prepare data for seasonal prediction
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Features for prediction
    features = ['day_of_week', 'month', 'day']
    predictions = {}
    last_date = df['date'].max()
    
    for category in df['category'].unique():
        try:
            # Filter data for this category
            category_df = df[df['category'] == category].copy()
            if len(category_df) < 2:  # Skip if not enough data
                continue
                
            X = category_df[features]
            y = category_df['daily_revenue']
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Generate future dates
            future_dates = [last_date + timedelta(days=x) for x in range(1, periods + 1)]
            future_df = pd.DataFrame({
                'date': future_dates,
                'day_of_week': [d.dayofweek for d in future_dates],
                'month': [d.month for d in future_dates],
                'day': [d.day for d in future_dates]
            })
            
            # Predict
            X_future = future_df[features]
            X_future_scaled = scaler.transform(X_future)
            predictions[category] = pd.DataFrame({
                'date': future_dates,
                'predicted_revenue': model.predict(X_future_scaled)
            })
        except Exception as e:
            st.warning(f"Could not generate predictions for {category}: {str(e)}")
            continue
    
    return predictions

def show_admin_dashboard(df):
    st.title("📊 Admin Dashboard")
    
    # Add month column for seasonal analysis
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("📌 Filters")
        selected_area = st.multiselect("Select Areas", df['area'].unique(), default=df['area'].unique())
        selected_category = st.multiselect("Select Categories", df['category'].unique(), default=df['category'].unique())
        
        default_end_date = df['date'].max().date()
        default_start_date = default_end_date - timedelta(days=30)
        
        date_range = st.date_input(
            "Select Date Range",
            value=[default_start_date, default_end_date],
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = default_start_date
            end_date = default_end_date
    
    # Filter data
    filtered_df = df[
        (df['area'].isin(selected_area)) &
        (df['category'].isin(selected_category)) &
        (df['date'].dt.date >= start_date) &
        (df['date'].dt.date <= end_date)
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Overview metrics
    st.markdown('<p class="section-header">Overview Metrics</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics for the selected period
    total_revenue = filtered_df['daily_revenue'].sum()
    avg_footfall = filtered_df['customer_footfall'].mean()
    total_stores = filtered_df['store_id'].nunique()
    avg_margin = filtered_df['margin'].mean()
    
    # Calculate previous period metrics
    prev_start = start_date - timedelta(days=(end_date - start_date).days)
    prev_end = start_date - timedelta(days=1)
    
    prev_df = df[
        (df['area'].isin(selected_area)) &
        (df['category'].isin(selected_category)) &
        (df['date'].dt.date >= prev_start) &
        (df['date'].dt.date <= prev_end)
    ].copy()
    
    prev_revenue = prev_df['daily_revenue'].sum() if not prev_df.empty else 0
    prev_footfall = prev_df['customer_footfall'].mean() if not prev_df.empty else 0
    
    revenue_change = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue != 0 else 0
    footfall_change = ((avg_footfall - prev_footfall) / prev_footfall * 100) if prev_footfall != 0 else 0
    
    # Display current period vs previous period
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Period Comparison")
    st.sidebar.markdown(f"**Current Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**Previous Period:** {prev_start.strftime('%Y-%m-%d')} to {prev_end.strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**Revenue Change:** {'↑' if revenue_change > 0 else '↓'} {abs(revenue_change):.1f}%")
    
    metrics = [
        ("Total Revenue", f"₹{total_revenue:,.0f}", f"{revenue_change:+.1f}%", "💰"),
        ("Avg. Daily Footfall", f"{avg_footfall:,.0f}", f"{footfall_change:+.1f}%", "👥"),
        ("Total Stores", f"{total_stores}", "", "🏪"),
        ("Avg. Margin", f"{avg_margin:.1f}%", "", "📊")
    ]
    
    for col, (title, value, change, emoji) in zip([col1, col2, col3, col4], metrics):
        with col:
            # Only include the span tag if there's a change value
            span_html = f'''
                <span style="color: {'green' if float(change.strip('%+') or 0) > 0 else 'red'}; font-size: 0.9rem;">
                    {change}
                </span>''' if change else ''
            
            st.markdown(f'''
            <div class="custom-metric">
                <h3>{emoji} {title}</h3>
                <p>{value}</p>{span_html}
            </div>
            ''', unsafe_allow_html=True)

    # Performance Analysis
    st.markdown('<p class="section-header">Performance Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Revenue Analysis", "🎯 Category Performance", "📍 Area Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Daily Revenue Trend
            daily_revenue = filtered_df.groupby('date')['daily_revenue'].sum().reset_index()
            fig = px.line(
                daily_revenue,
                x='date',
                y='daily_revenue',
                title="Daily Revenue Trend",
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate revenue trend and add interpretation
            revenue_trend = daily_revenue['daily_revenue'].pct_change().mean()
            trend_text = "increasing" if revenue_trend > 0 else "decreasing"
            max_revenue = daily_revenue['daily_revenue'].max()
            max_date = daily_revenue[daily_revenue['daily_revenue'] == max_revenue]['date'].iloc[0].strftime('%Y-%m-%d')
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Revenue shows a {trend_text} trend with peak revenue of ₹{max_revenue:,.0f} on {max_date}.</div>", unsafe_allow_html=True)
        
        with col2:
            # Day of Week Analysis
            dow_revenue = filtered_df.groupby('day_of_week')['daily_revenue'].mean().reset_index()
            dow_revenue['day_name'] = dow_revenue['day_of_week'].map({
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            })
            fig = px.bar(
                dow_revenue,
                x='day_name',
                y='daily_revenue',
                title="Average Revenue by Day of Week",
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate revenue trend
            revenue_trend = daily_revenue['daily_revenue'].pct_change().mean()
            trend_text = "increasing" if revenue_trend > 0 else "decreasing"
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Daily revenue shows a {trend_text} trend over the selected period.</div>", unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # Category Performance Matrix
            category_metrics = filtered_df.groupby('category').agg({
                'daily_revenue': 'sum',
                'margin': 'mean',
                'actual_demand': 'sum',
                'stock': 'mean'
            }).reset_index()
            
            fig = px.scatter(
                category_metrics,
                x='margin',
                y='daily_revenue',
                size='actual_demand',
                color='category',
                title="Category Performance Matrix",
                labels={
                    'margin': 'Average Margin (%)',
                    'daily_revenue': 'Total Revenue (₹)',
                    'actual_demand': 'Total Demand'
                },
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation for category performance
            top_category = category_metrics.loc[category_metrics['daily_revenue'].idxmax()]
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📊 {top_category['category']} leads in revenue (₹{top_category['daily_revenue']:,.0f}) with {top_category['margin']:.1f}% margin.</div>", unsafe_allow_html=True)
        
        with col2:
            # Category Growth Rates
            category_growth = filtered_df.groupby(['category', 'week']).agg({
                'daily_revenue': 'sum'
            }).reset_index()
            
            fig = px.line(
                category_growth,
                x='week',
                y='daily_revenue',
                color='category',
                title="Weekly Category Revenue Trends",
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate revenue trend
            revenue_trend = daily_revenue['daily_revenue'].pct_change().mean()
            trend_text = "increasing" if revenue_trend > 0 else "decreasing"
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Daily revenue shows a {trend_text} trend over the selected period.</div>", unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Area Performance Comparison
            area_metrics = filtered_df.groupby('area').agg({
                'daily_revenue': 'sum',
                'customer_footfall': 'mean',
                'margin': 'mean'
            }).reset_index()
            
            fig = px.scatter(
                area_metrics,
                x='customer_footfall',
                y='daily_revenue',
                size='margin',
                color='area',
                title="Area Performance Comparison",
                labels={
                    'customer_footfall': 'Average Daily Footfall',
                    'daily_revenue': 'Total Revenue (₹)',
                    'margin': 'Average Margin (%)'
                },
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation for area performance
            top_area = area_metrics.loc[area_metrics['daily_revenue'].idxmax()]
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📍 {top_area['area']} generates highest revenue (₹{top_area['daily_revenue']:,.0f}) with {top_area['customer_footfall']:.0f} daily customers.</div>", unsafe_allow_html=True)
        
        with col2:
            # Area-Category Heatmap
            area_category = pd.pivot_table(
                filtered_df,
                values='daily_revenue',
                index='area',
                columns='category',
                aggfunc='sum'
            )
            
            fig = px.imshow(
                area_category,
                title="Area-Category Revenue Heatmap",
                labels=dict(x="Category", y="Area", color="Revenue"),
                aspect="auto",
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation for area-category heatmap
            max_revenue = area_category.max().max()
            max_area = area_category.max(axis=1).idxmax()
            max_category = area_category.max(axis=0).idxmax()
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>🔥 Highest revenue combination: {max_area} - {max_category} (₹{max_revenue:,.0f})</div>", unsafe_allow_html=True)
    
    # Inventory Analysis
    st.markdown('<p class="section-header">Inventory Insights</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Stock vs Demand Analysis
        inventory_metrics = filtered_df.groupby('category').agg({
            'stock': 'mean',
            'actual_demand': 'mean'
        }).reset_index()
        
        inventory_metrics['stock_coverage_days'] = inventory_metrics['stock'] / inventory_metrics['actual_demand']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Average Stock',
            x=inventory_metrics['category'],
            y=inventory_metrics['stock'],
            marker_color='#1e3c72'
        ))
        fig.add_trace(go.Bar(
            name='Average Daily Demand',
            x=inventory_metrics['category'],
            y=inventory_metrics['actual_demand'],
            marker_color='#2a5298'
        ))
        
        fig.update_layout(
            title="Stock vs Demand by Category",
            barmode='group',
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation for stock vs demand
        highest_demand = inventory_metrics.loc[inventory_metrics['actual_demand'].idxmax()]
        st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📦 {highest_demand['category']} has highest daily demand ({highest_demand['actual_demand']:.0f} units) with {highest_demand['stock']:.0f} units in stock.</div>", unsafe_allow_html=True)
    
    with col2:
        # Stock Coverage Analysis
        fig = px.bar(
            inventory_metrics,
            x='category',
            y='stock_coverage_days',
            title="Stock Coverage Days by Category",
            labels={'stock_coverage_days': 'Days of Stock Coverage'},
            template='plotly_white'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Minimum Target (7 days)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Maximum Target (30 days)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation for stock coverage
        low_coverage = inventory_metrics[inventory_metrics['stock_coverage_days'] < 7]
        high_coverage = inventory_metrics[inventory_metrics['stock_coverage_days'] > 30]
        if not low_coverage.empty:
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>⚠️ {', '.join(low_coverage['category'])} have less than 7 days of stock coverage.</div>", unsafe_allow_html=True)
        elif not high_coverage.empty:
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>ℹ️ {', '.join(high_coverage['category'])} have more than 30 days of stock coverage.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; font-size: 0.9rem; color: #666;'>✅ All categories have optimal stock coverage (7-30 days).</div>", unsafe_allow_html=True)

    # Add Seasonal Predictions
    st.markdown('<p class="section-header">Seasonal Predictions</p>', unsafe_allow_html=True)
    predictions = predict_seasonal_trends(filtered_df)
    
    tab1, tab2 = st.tabs(["📈 Revenue Forecast", "📊 Category Trends"])
    
    with tab1:
        # Combine all category predictions
        all_predictions = pd.concat([
            pred.assign(category=cat) for cat, pred in predictions.items()
        ])
        
        fig = px.line(
            all_predictions,
            x='date',
            y='predicted_revenue',
            color='category',
            title="30-Day Revenue Forecast by Category",
            labels={'predicted_revenue': 'Predicted Revenue (₹)', 'date': 'Date'},
            template='plotly_white'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Show category-wise growth trends
        growth_data = []
        for category, pred_df in predictions.items():
            current_revenue = filtered_df[filtered_df['category'] == category]['daily_revenue'].mean()
            predicted_revenue = pred_df['predicted_revenue'].mean()
            growth = ((predicted_revenue - current_revenue) / current_revenue * 100)
            growth_data.append({
                'category': category,
                'growth': growth,
                'current_revenue': current_revenue,
                'predicted_revenue': predicted_revenue
            })
        
        growth_df = pd.DataFrame(growth_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                growth_df,
                x='category',
                y='growth',
                title="Predicted Growth Rate by Category (%)",
                labels={'growth': 'Growth Rate (%)', 'category': 'Category'},
                template='plotly_white'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate revenue trend
            revenue_trend = daily_revenue['daily_revenue'].pct_change().mean()
            trend_text = "increasing" if revenue_trend > 0 else "decreasing"
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Daily revenue shows a {trend_text} trend over the selected period.</div>", unsafe_allow_html=True)
        
        with col2:
            # Show recommendations based on predictions
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                <h3 style="color: #1e3c72; margin-bottom: 1rem;">🎯 Growth Opportunities</h3>
            """, unsafe_allow_html=True)
            
            # High growth categories
            high_growth = growth_df[growth_df['growth'] > growth_df['growth'].mean()]
            if not high_growth.empty:
                st.markdown("**High Growth Categories:**")
                for _, row in high_growth.iterrows():
                    st.markdown(f"- **{row['category']}**: Expected growth of {row['growth']:.1f}%")
            
            # Categories needing attention
            low_growth = growth_df[growth_df['growth'] < 0]
            if not low_growth.empty:
                st.markdown("\n**Categories Needing Attention:**")
                for _, row in low_growth.iterrows():
                    st.markdown(f"- **{row['category']}**: Expected decline of {abs(row['growth']):.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_store_dashboard(df):
    # Get store information
    store_id = st.session_state['store_id']
    store_area = USERS[st.session_state['username']]['area']
    
    st.title(f"📊 Store Dashboard: {store_area} ({store_id})")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Inventory Analysis", "Sales Trends", "Data Management"])
    
    # Filter data for this store
    store_df = df[df['store_id'] == store_id].copy()
    
    # Add logout button in sidebar
    with st.sidebar:
        if st.button("Logout", key="logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Overview Tab
    with tab1:
        st.subheader("Store Performance Overview")
        
        # Date filters for overview
        with st.sidebar:
            st.subheader("📅 Date Range")
            
            default_end_date = df['date'].max().date()
            default_start_date = default_end_date - timedelta(days=30)
            
            date_range = st.date_input(
                "Select Date Range",
                value=[default_start_date, default_end_date],
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = default_start_date
                end_date = default_end_date
        
        # Filter by date
        filtered_df = store_df[
            (store_df['date'].dt.date >= start_date) &
            (store_df['date'].dt.date <= end_date)
        ]
        
        # Calculate metrics
        total_revenue = filtered_df['daily_revenue'].sum()
        avg_daily_revenue = filtered_df.groupby('date')['daily_revenue'].sum().mean()
        total_customers = filtered_df['customer_footfall'].sum()
        avg_margin = filtered_df['margin'].mean()
        
        # Previous period for comparison
        prev_start = start_date - timedelta(days=(end_date - start_date).days)
        prev_end = start_date - timedelta(days=1)
        
        prev_df = store_df[
            (store_df['date'].dt.date >= prev_start) &
            (store_df['date'].dt.date <= prev_end)
        ]
        
        prev_revenue = prev_df['daily_revenue'].sum() if not prev_df.empty else 0
        revenue_change = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue != 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Revenue", 
                f"₹{total_revenue:,.0f}", 
                f"{revenue_change:+.1f}%"
            )
        
        with col2:
            st.metric(
                "Avg. Daily Revenue", 
                f"₹{avg_daily_revenue:,.0f}"
            )
        
        with col3:
            st.metric(
                "Total Customers", 
                f"{total_customers:,.0f}"
            )
        
        with col4:
            st.metric(
                "Avg. Margin", 
                f"{avg_margin:.1f}%"
            )
        
        # Revenue trend
        st.subheader("Revenue Trend")
        daily_revenue = filtered_df.groupby('date')['daily_revenue'].sum().reset_index()
        
        fig = px.line(
            daily_revenue, 
            x='date', 
            y='daily_revenue',
            labels={'daily_revenue': 'Daily Revenue (₹)', 'date': 'Date'},
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation for store revenue trend
        revenue_trend = daily_revenue['daily_revenue'].pct_change().mean()
        trend_text = "increasing" if revenue_trend > 0 else "decreasing"
        max_revenue = daily_revenue['daily_revenue'].max()
        max_date = daily_revenue[daily_revenue['daily_revenue'] == max_revenue]['date'].iloc[0].strftime('%Y-%m-%d')
        st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Store revenue shows a {trend_text} trend with peak revenue of ₹{max_revenue:,.0f} on {max_date}.</div>", unsafe_allow_html=True)
        
        # Category performance
        st.subheader("Category Performance")
        
        category_perf = filtered_df.groupby('category').agg({
            'daily_revenue': 'sum',
            'customer_footfall': 'sum',
            'margin': 'mean'
        }).reset_index()
        
        category_perf = category_perf.sort_values('daily_revenue', ascending=False)
        
        fig = px.bar(
            category_perf,
            x='category',
            y='daily_revenue',
            color='margin',
            labels={'daily_revenue': 'Revenue (₹)', 'category': 'Category', 'margin': 'Margin (%)'},
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation for store category performance
        top_category = category_perf.iloc[0]
        st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📊 {top_category['category']} is the top performer with ₹{top_category['daily_revenue']:,.0f} revenue and {top_category['margin']:.1f}% margin.</div>", unsafe_allow_html=True)
    
    # Inventory Analysis Tab
    with tab2:
        st.subheader("Inventory Analysis")
        
        # Category filter for inventory
        selected_category = st.selectbox(
            "Select Category",
            options=store_df['category'].unique(),
            key="inventory_category"
        )
        
        # Filter by category
        category_df = store_df[store_df['category'] == selected_category].copy()
        
        # Calculate inventory metrics
        current_stock = category_df.iloc[-1]['stock'] if not category_df.empty else 0
        avg_daily_sales = category_df['actual_demand'].mean() if not category_df.empty else 0
        days_of_supply = current_stock / avg_daily_sales if avg_daily_sales > 0 else 0
        
        # Display inventory metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Stock", 
                f"{current_stock:,.0f} units"
            )
        
        with col2:
            st.metric(
                "Avg. Daily Sales", 
                f"{avg_daily_sales:.1f} units"
            )
        
        with col3:
            supply_color = "normal"
            if days_of_supply < 7:
                supply_color = "off"
            elif days_of_supply > 30:
                supply_color = "inverse"
                
            st.metric(
                "Days of Supply", 
                f"{days_of_supply:.1f} days",
                delta_color=supply_color
            )
        
        # Stock vs Sales trend
        st.subheader("Stock vs Sales Trend")
        
        stock_sales = category_df[['date', 'stock', 'actual_demand']].copy()
        stock_sales.rename(columns={'actual_demand': 'daily_sales'}, inplace=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=stock_sales['date'], 
                y=stock_sales['stock'],
                name="Stock Level",
                line=dict(color='blue')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(
                x=stock_sales['date'], 
                y=stock_sales['daily_sales'],
                name="Daily Sales",
                marker_color='rgba(0, 128, 0, 0.5)'
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="Stock Level vs Daily Sales",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Stock Level (units)", secondary_y=False)
        fig.update_yaxes(title_text="Daily Sales (units)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation for stock vs sales trend
        avg_daily_sales = stock_sales['daily_sales'].mean()
        current_stock = stock_sales['stock'].iloc[-1]
        days_remaining = current_stock / avg_daily_sales if avg_daily_sales > 0 else 0
        st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📦 Current stock of {current_stock:.0f} units covers {days_remaining:.1f} days of average sales ({avg_daily_sales:.1f} units/day).</div>", unsafe_allow_html=True)
        
        # Inventory Optimization
        st.subheader("Inventory Optimization")
        
        # Load the demand predictor
        predictor = DemandPredictor()
        try:
            try:
                predictor.load_models()
                models_loaded = True
            except Exception as model_error:
                st.warning(f"Could not load prediction models: {str(model_error)}. Using simple forecasting instead.")
                models_loaded = False
            
            # Prepare data for prediction
            last_date = category_df['date'].max()
            future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
            
            # Create prediction dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'store_id': store_id,
                'category': selected_category,
                'area': store_area,
                # Add other required columns with default values
                'seasonal_factor': 1.0,
                'festival_boost': 0.0
            })
            
            # Add other columns that might be needed for prediction
            for col in ['store_size', 'product_id', 'subcategory', 'income_level', 
                       'credit_limit', 'customer_footfall', 'discount', 'stock', 
                       'margin', 'base_price', 'competitor_price', 'storage_capacity',
                       'stock_days_remaining', 'is_urban', 'is_promotional_period']:
                if col in category_df.columns:
                    future_df[col] = category_df[col].iloc[-1]
                else:
                    future_df[col] = 0  # Default value
            
            # Make predictions
            if models_loaded:
                predictions = predictor.predict(future_df)
            else:
                # Simple linear regression as fallback
                if len(category_df) > 5:
                    X = np.array(range(len(category_df))).reshape(-1, 1)
                    y = category_df['actual_demand'].values
                    model = LinearRegression()
                    model.fit(X, y)
                    future_X = np.array(range(len(category_df), len(category_df) + 30)).reshape(-1, 1)
                    predictions = model.predict(future_X)
                else:
                    # If not enough data, just use the average
                    avg_demand = category_df['actual_demand'].mean() if not category_df.empty else 0
                    predictions = np.array([avg_demand] * 30)
            
            # Create dataframe with predictions
            prediction_df = pd.DataFrame({
                'date': future_dates,
                'predicted_demand': predictions
            })
            
            # Calculate optimal order quantity
            lead_time = 2  # Assumed lead time in days
            safety_stock = prediction_df['predicted_demand'].std() * 1.5  # Safety stock based on demand variability
            
            # Calculate Economic Order Quantity (EOQ)
            annual_demand = avg_daily_sales * 365
            holding_cost_percent = 0.2  # 20% of item cost
            ordering_cost = 500  # Fixed cost per order in ₹
            item_cost = category_df['daily_revenue'].sum() / category_df['actual_demand'].sum() if category_df['actual_demand'].sum() > 0 else 100
            
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / (item_cost * holding_cost_percent))
            
            # Calculate reorder point
            reorder_point = (avg_daily_sales * lead_time) + safety_stock
            
            # Display optimization metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Economic Order Quantity", 
                    f"{eoq:.0f} units"
                )
            
            with col2:
                st.metric(
                    "Reorder Point", 
                    f"{reorder_point:.0f} units"
                )
            
            with col3:
                st.metric(
                    "Safety Stock", 
                    f"{safety_stock:.0f} units"
                )
            
            # Plot demand forecast
            st.subheader("Demand Forecast (Next 30 Days)")
            
            fig = px.line(
                prediction_df, 
                x='date', 
                y='predicted_demand',
                labels={'predicted_demand': 'Predicted Daily Demand', 'date': 'Date'},
                template='plotly_white'
            )
            
            # Add current stock level line
            fig.add_hline(
                y=current_stock, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Current Stock Level",
                annotation_position="bottom right"
            )
            
            # Add reorder point line
            fig.add_hline(
                y=reorder_point, 
                line_dash="dot", 
                line_color="orange",
                annotation_text="Reorder Point",
                annotation_position="bottom left"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation for demand forecast
            avg_predicted_demand = prediction_df['predicted_demand'].mean()
            demand_change = ((avg_predicted_demand - avg_daily_sales) / avg_daily_sales * 100) if avg_daily_sales > 0 else 0
            trend_text = "increase" if demand_change > 0 else "decrease"
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Forecast predicts {abs(demand_change):.1f}% {trend_text} in daily demand (avg. {avg_predicted_demand:.1f} units) over next 30 days.</div>", unsafe_allow_html=True)
            
            # Order recommendation
            if current_stock <= reorder_point:
                st.warning(f"⚠️ Stock level is below or at reorder point. Consider placing an order of {eoq:.0f} units.")
            else:
                days_until_reorder = (current_stock - reorder_point) / avg_daily_sales if avg_daily_sales > 0 else float('inf')
                st.info(f"ℹ️ Stock level is adequate. Estimated {days_until_reorder:.1f} days until reorder point is reached.")
                
        except Exception as e:
            st.error(f"Error loading prediction models: {str(e)}")
    
    # Sales Trends Tab
    with tab3:
        st.subheader("Sales Trends Analysis")
        
        # Time period selector
        period_options = ["Daily", "Weekly", "Monthly"]
        selected_period = st.selectbox("Select Time Period", options=period_options, key="trend_period")
        
        # Category selector
        selected_categories = st.multiselect(
            "Select Categories",
            options=store_df['category'].unique(),
            default=store_df['category'].unique()[0:3],
            key="trend_categories"
        )
        
        # Filter by selected categories
        if selected_categories:
            trend_df = store_df[store_df['category'].isin(selected_categories)].copy()
        else:
            trend_df = store_df.copy()
        
        # Aggregate based on selected period
        if selected_period == "Daily":
            trend_data = trend_df.groupby(['date', 'category']).agg({
                'daily_revenue': 'sum',
                'actual_demand': 'sum',
                'customer_footfall': 'mean'
            }).reset_index()
            x_axis = 'date'
        elif selected_period == "Weekly":
            trend_df['week'] = trend_df['date'].dt.isocalendar().week
            trend_df['year'] = trend_df['date'].dt.year
            trend_df['week_year'] = trend_df['year'].astype(str) + "-W" + trend_df['week'].astype(str).str.zfill(2)
            
            trend_data = trend_df.groupby(['week_year', 'category']).agg({
                'daily_revenue': 'sum',
                'actual_demand': 'sum',
                'customer_footfall': 'mean'
            }).reset_index()
            x_axis = 'week_year'
        else:  # Monthly
            trend_df['month'] = trend_df['date'].dt.month
            trend_df['year'] = trend_df['date'].dt.year
            trend_df['month_year'] = trend_df['year'].astype(str) + "-" + trend_df['month'].astype(str).str.zfill(2)
            
            trend_data = trend_df.groupby(['month_year', 'category']).agg({
                'daily_revenue': 'sum',
                'actual_demand': 'sum',
                'customer_footfall': 'mean'
            }).reset_index()
            x_axis = 'month_year'
        
        # Plot revenue trends
        st.subheader("Revenue Trends by Category")
        
        fig = px.line(
            trend_data, 
            x=x_axis, 
            y='daily_revenue',
            color='category',
            labels={'daily_revenue': 'Revenue (₹)', x_axis: 'Time Period'},
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate revenue growth trend
        revenue_trend = trend_data.groupby(x_axis)['daily_revenue'].sum().pct_change().mean()
        trend_text = "increasing" if revenue_trend > 0 else "decreasing"
        st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Revenue shows a {trend_text} trend across categories over the selected period.</div>", unsafe_allow_html=True)
        
        # Plot sales trends
        st.subheader("Sales Volume Trends by Category")
        
        fig = px.line(
            trend_data, 
            x=x_axis, 
            y='actual_demand',
            color='category',
            labels={'actual_demand': 'Sales Volume (units)', x_axis: 'Time Period'},
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate sales volume growth trend
        sales_trend = trend_data.groupby(x_axis)['actual_demand'].sum().pct_change().mean()
        trend_text = "increasing" if sales_trend > 0 else "decreasing"
        st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Sales volume shows a {trend_text} trend across categories over the selected period.</div>", unsafe_allow_html=True)
        
        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        
        # Add day of week and month columns
        trend_df['day_of_week'] = trend_df['date'].dt.dayofweek
        trend_df['day_name'] = trend_df['date'].dt.day_name()
        trend_df['month'] = trend_df['date'].dt.month
        trend_df['month_name'] = trend_df['date'].dt.month_name()
        
        # Day of week patterns
        daily_pattern = trend_df.groupby('day_of_week').agg({
            'daily_revenue': 'mean',
            'customer_footfall': 'mean'
        }).reset_index()
        
        daily_pattern['day_name'] = daily_pattern['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        # Sort by day of week
        daily_pattern = daily_pattern.sort_values('day_of_week')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Patterns")
            
            fig = px.bar(
                daily_pattern,
                x='day_name',
                y='daily_revenue',
                labels={'daily_revenue': 'Avg. Daily Revenue (₹)', 'day_name': 'Day of Week'},
                template='plotly_white'
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation for daily patterns
            best_day = daily_pattern.loc[daily_pattern['daily_revenue'].idxmax()]
            worst_day = daily_pattern.loc[daily_pattern['daily_revenue'].idxmin()]
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📅 {best_day['day_name']} generates highest revenue (₹{best_day['daily_revenue']:,.0f}), while {worst_day['day_name']} is lowest (₹{worst_day['daily_revenue']:,.0f}).</div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Customer Footfall by Day")
            
            fig = px.bar(
                daily_pattern,
                x='day_name',
                y='customer_footfall',
                labels={'customer_footfall': 'Avg. Customer Footfall', 'day_name': 'Day of Week'},
                template='plotly_white'
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate revenue trend
            revenue_trend = daily_revenue['daily_revenue'].pct_change().mean()
            trend_text = "increasing" if revenue_trend > 0 else "decreasing"
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📈 Daily revenue shows a {trend_text} trend over the selected period.</div>", unsafe_allow_html=True)
        
        # Monthly patterns
        monthly_pattern = trend_df.groupby('month').agg({
            'daily_revenue': 'mean',
            'customer_footfall': 'mean'
        }).reset_index()
        
        monthly_pattern['month_name'] = monthly_pattern['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        # Sort by month
        monthly_pattern = monthly_pattern.sort_values('month')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Revenue Patterns")
            
            fig = px.line(
                monthly_pattern,
                x='month_name',
                y='daily_revenue',
                labels={'daily_revenue': 'Avg. Daily Revenue (₹)', 'month_name': 'Month'},
                template='plotly_white'
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation for monthly patterns
            best_month = monthly_pattern.loc[monthly_pattern['daily_revenue'].idxmax()]
            worst_month = monthly_pattern.loc[monthly_pattern['daily_revenue'].idxmin()]
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>📅 {best_month['month_name']} shows highest average revenue (₹{best_month['daily_revenue']:,.0f}), while {worst_month['month_name']} is lowest (₹{worst_month['daily_revenue']:,.0f}).</div>", unsafe_allow_html=True)
    
    # Data Management Tab
    with tab4:
        # Add the data upload section
        data_upload_section(store_id)

def show_landing_page():
    st.markdown("""
        <style>
        body, .stApp {
            background: linear-gradient(135deg, #e0e7ff 0%, #f8fafc 100%);
        }
        .landing-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 1rem;
        }
        .hero-section {
            text-align: center;
            padding: 3rem 1.5rem 2rem 1.5rem;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(4px);
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            position: relative;
        }
        .hero-section .highlight {
            color: #ffd700;
            font-weight: 700;
        }
        .info-section {
            background: rgba(255,255,255,0.9);
            border-radius: 16px;
            box-shadow: 0 4px 24px 0 rgba(30,60,114,0.08);
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            font-size: 1.08rem;
            color: #1a202c;
        }
        .info-section h3 {
            color: #2a5298;
            margin-bottom: 0.5rem;
        }
        .info-section .accent {
            color: #1e3c72;
            font-weight: 600;
        }
        .warning-box {
            background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%);
            color: #7c2d12;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            font-weight: 500;
            box-shadow: 0 2px 8px 0 rgba(252,182,159,0.12);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .warning-box .icon {
            font-size: 1.5rem;
        }
        .section-title {
            color: #1e3c72 !important;
            margin-bottom: 1rem;
            position: relative;
            display: inline-block;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin: 1.5rem 0;
            position: relative;
        }
        .feature-card {
            background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(30,60,114,0.07);
            margin-bottom: 0;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(30, 60, 114, 0.08);
            display: flex;
            align-items: center;
            gap: 1rem;
            color: #1a202c;
        }
        .feature-card:hover {
            transform: translateY(-3px) scale(1.03);
            box-shadow: 0 8px 24px rgba(30,60,114,0.13);
            border-color: #2a5298;
        }
        .feature-icon {
            font-size: 2.5rem;
            color: #fff;
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            border-radius: 12px;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(30,60,114,0.10);
        }
        .feature-content {
            flex-grow: 1;
        }
        .step-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin: 2rem 0;
            position: relative;
        }
        .step-box {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(30,60,114,0.07);
            text-align: left;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(30, 60, 114, 0.08);
            display: flex;
            flex-direction: column;
            gap: 1rem;
            color: #1a202c;
        }
        .step-box:hover {
            transform: translateY(-3px) scale(1.03);
            box-shadow: 0 8px 24px rgba(30,60,114,0.13);
            border-color: #2a5298;
        }
        .step-number {
            font-size: 1.5rem;
            font-weight: bold;
            color: #fff;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            border-radius: 50%;
            margin-bottom: 0.5rem;
        }
        .step-content {
            flex-grow: 1;
        }
        .cta-container {
            text-align: center;
            margin: 2.5rem 0 1rem 0;
            padding: 1rem;
        }
        .cta-button {
            display: inline-block;
            padding: 1.1rem 3.5rem;
            background: linear-gradient(135deg, #ffb347 0%, #1e3c72 100%);
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.2rem;
            transition: all 0.3s cubic-bezier(.4,2,.6,1);
            position: relative;
            overflow: hidden;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 16px rgba(30,60,114,0.13);
            letter-spacing: 0.5px;
        }
        .cta-button:hover {
            transform: translateY(-2px) scale(1.04);
            box-shadow: 0 8px 32px rgba(255,179,71,0.18);
            background: linear-gradient(135deg, #1e3c72 0%, #ffb347 100%);
        }
        @media (max-width: 768px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }
            .step-container {
                grid-template-columns: 1fr;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 style="font-size: 2.7rem; margin-bottom: 0.5rem;">Welcome to <span class="highlight">Stockify</span></h1>
            <h2 style="font-size: 1.3rem; font-weight: normal; margin-bottom: 0.5rem;">Smarter Demand Forecasting for Your Store</h2>
            <div style="font-size: 1.1rem; margin-top: 0.5rem; color: #ffd700;">
                <b>Reduce stockouts</b> · <b>Boost profits</b> · <b>Delight your customers</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Why Demand Forecasting Section
    st.markdown("""
        <div class="landing-container">
            <div class="info-section">
                <h3>Why Demand Forecasting?</h3>
                <p>
                    <span class="accent">Demand forecasting</span> is the backbone of successful retail. Without it, stores risk <span class="accent">overstocking</span> (wasting money and space) or <span class="accent">stockouts</span> (losing sales and customers).<br><br>
                    <b>Stockify</b> uses AI to help you predict what your customers will want—so you always have the right products, at the right time.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Who Needs Stockify Section
    st.markdown("""
        <div class="landing-container">
            <div class="info-section">
                <h3>Who Needs Stockify?</h3>
                <ul style="margin: 0 0 0 1.2rem; color: #1a202c;">
                    <li>Retailers who want to <span class="accent">maximize sales</span> and <span class="accent">minimize waste</span></li>
                    <li>Store managers aiming for <span class="accent">data-driven decisions</span></li>
                    <li>Chains/franchises seeking <span class="accent">consistency</span> across locations</li>
                    <li>Anyone tired of <span class="accent">guesswork</span> in inventory planning</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Warning/Info Box
    st.markdown("""
        <div class="landing-container">
            <div class="warning-box"><span class="icon">⚠️</span>
                <span>Not forecasting demand can lead to <b>lost sales</b>, <b>excess inventory</b>, and <b>cash flow problems</b>. Don't leave your business to chance!</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # What We Offer Section
    st.markdown("""
        <div class="landing-container">
            <h2 class="section-title">What We Offer</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon" style="background: linear-gradient(135deg, #ffb347 0%, #ffcc80 100%); color: #1e3c72;">🤖</div>
                    <div class="feature-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">AI-Powered Predictions</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Accurate demand forecasts using advanced machine learning algorithms</p>
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon" style="background: linear-gradient(135deg, #6dd5ed 0%, #2193b0 100%); color: #fff;">📊</div>
                    <div class="feature-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">Inventory Optimization</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Smart stock level recommendations to minimize costs and maximize sales</p>
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon" style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%); color: #fff;">📈</div>
                    <div class="feature-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">Real-time Insights</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Live sales analytics and performance metrics at your fingertips</p>
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon" style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); color: #1e3c72;">🎯</div>
                    <div class="feature-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">User-friendly Dashboards</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Intuitive interfaces for both store managers and administrators</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("""
        <div class="landing-container">
            <h2 class="section-title">How It Works</h2>
            <div class="step-container">
                <div class="step-box">
                    <div class="step-number" style="background: linear-gradient(135deg, #ffb347 0%, #ffcc80 100%);">1</div>
                    <div class="step-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">Upload Data</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Simply upload your store's historical sales data</p>
                    </div>
                </div>
                <div class="step-box">
                    <div class="step-number" style="background: linear-gradient(135deg, #6dd5ed 0%, #2193b0 100%);">2</div>
                    <div class="step-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">Get Forecasts</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Our AI analyzes patterns and generates predictions</p>
                    </div>
                </div>
                <div class="step-box">
                    <div class="step-number" style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);">3</div>
                    <div class="step-content">
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1e3c72;">Make Decisions</h3>
                        <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">Use insights to optimize inventory and boost sales</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def main():
    # Initialize database
    engine = init_db()
    initialize_stores(engine, USERS)
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'show_landing' not in st.session_state:
        st.session_state['show_landing'] = True
    
    if st.session_state['show_landing']:
        show_landing_page()
        # Hide the Streamlit button visually, but keep it for JS click
        st.markdown("""
            <style>
            div.stButton > button#continue_to_login {
                display: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Continue to Login", key="continue_to_login"):
                st.session_state['show_landing'] = False
                st.rerun()
    elif not st.session_state['logged_in']:
        login()
    else:
        # Load data
        if st.session_state['role'] == 'admin':
            # Load historical data for admin dashboard
            df = load_admin_data()
            if df is not None:
                show_admin_dashboard(df)
        else:
            # For store users, check if they have uploaded data
            store_id = st.session_state['store_id']
            store_data = get_store_data(store_id)
            
            if store_data is not None:
                # Use store's uploaded data
                show_store_dashboard(store_data)
            else:
                # Use retail_data.csv filtered for this store
                df = load_store_data()
                if df is not None:
                    store_df = df[df['store_id'] == store_id].copy()
                    show_store_dashboard(store_df)

if __name__ == "__main__":
    main()
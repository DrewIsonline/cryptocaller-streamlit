import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Optional

# Import our custom modules
from src.market_data import MarketDataManager
from src.trading_bot import CryptoTradingBot
from src.risk_management import RiskManager
from src.portfolio_manager import PortfolioManager

# Page configuration
st.set_page_config(
    page_title="CryptoCaller - Advanced Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'market_data_manager' not in st.session_state:
    st.session_state.market_data_manager = MarketDataManager()
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()
if 'selected_exchanges' not in st.session_state:
    st.session_state.selected_exchanges = ['binance']
if 'trading_pairs' not in st.session_state:
    st.session_state.trading_pairs = ['BTC/USDT', 'ETH/USDT']

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 CryptoCaller - Advanced Trading Bot</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=CryptoCaller", width=200)
        
        st.markdown("### 📊 Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Trading Bot", "Market Analysis", "Portfolio", "Risk Management", "Settings"]
        )
        
        st.markdown("---")
        
        # Bot status
        st.markdown("### 🤖 Bot Status")
        if st.session_state.bot_running:
            st.markdown('<p class="status-running">🟢 Running</p>', unsafe_allow_html=True)
            if st.button("Stop Bot", type="secondary"):
                st.session_state.bot_running = False
                st.rerun()
        else:
            st.markdown('<p class="status-stopped">🔴 Stopped</p>', unsafe_allow_html=True)
            if st.button("Start Bot", type="primary"):
                st.session_state.bot_running = True
                st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Pairs", len(st.session_state.trading_pairs))
        with col2:
            st.metric("Exchanges", len(st.session_state.selected_exchanges))
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Trading Bot":
        show_trading_bot()
    elif page == "Market Analysis":
        show_market_analysis()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Risk Management":
        show_risk_management()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Display main dashboard"""
    st.header("📊 Trading Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Portfolio Value",
            value="$12,345.67",
            delta="$234.56 (1.9%)"
        )
    
    with col2:
        st.metric(
            label="24h P&L",
            value="$456.78",
            delta="3.7%"
        )
    
    with col3:
        st.metric(
            label="Active Positions",
            value="5",
            delta="2"
        )
    
    with col4:
        st.metric(
            label="Win Rate",
            value="68.5%",
            delta="2.1%"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Performance")
        # Sample data for portfolio performance
        dates = pd.date_range(start='2024-01-01', end='2024-07-24', freq='D')
        portfolio_values = np.cumsum(np.random.randn(len(dates)) * 50) + 10000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Asset Allocation")
        # Sample data for asset allocation
        assets = ['BTC', 'ETH', 'ADA', 'DOT', 'USDT']
        values = [35, 25, 15, 10, 15]
        
        fig = px.pie(
            values=values,
            names=assets,
            title="Current Asset Allocation"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades table
    st.subheader("Recent Trades")
    sample_trades = pd.DataFrame({
        'Time': ['2024-07-24 09:15:00', '2024-07-24 08:45:00', '2024-07-24 08:30:00'],
        'Pair': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        'Side': ['BUY', 'SELL', 'BUY'],
        'Amount': [0.1, 2.5, 1000],
        'Price': [67500, 3450, 0.45],
        'P&L': ['+$125.50', '+$87.30', '-$12.45'],
        'Status': ['Filled', 'Filled', 'Filled']
    })
    st.dataframe(sample_trades, use_container_width=True)

def show_trading_bot():
    """Display trading bot configuration and controls"""
    st.header("🤖 Trading Bot Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bot Settings")
        
        # Trading strategy selection
        strategy = st.selectbox(
            "Trading Strategy",
            ["Trend Following", "Mean Reversion", "Momentum", "Grid Trading"]
        )
        
        # Risk parameters
        max_position_size = st.slider("Max Position Size (%)", 1, 20, 5)
        stop_loss = st.slider("Stop Loss (%)", 1, 10, 3)
        take_profit = st.slider("Take Profit (%)", 5, 50, 15)
        
        # Trading pairs
        st.multiselect(
            "Trading Pairs",
            ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"],
            default=st.session_state.trading_pairs,
            key="trading_pairs_select"
        )
        
        # Update button
        if st.button("Update Configuration"):
            st.success("Configuration updated successfully!")
    
    with col2:
        st.subheader("Bot Performance")
        
        # Performance metrics
        metrics_data = {
            'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 'Avg Profit', 'Max Drawdown'],
            'Value': ['127', '87', '40', '68.5%', '$45.67', '8.2%']
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)
        
        # Bot logs
        st.subheader("Recent Bot Activity")
        logs = [
            "2024-07-24 09:15:23 - BUY signal generated for BTC/USDT",
            "2024-07-24 09:14:45 - Position opened: ETH/USDT",
            "2024-07-24 09:12:10 - Stop loss triggered: ADA/USDT",
            "2024-07-24 09:10:33 - Market analysis completed",
            "2024-07-24 09:08:15 - Risk check passed for all positions"
        ]
        
        for log in logs:
            st.text(log)

def show_market_analysis():
    """Display market analysis and charts"""
    st.header("📈 Market Analysis")
    
    # Symbol selector
    symbol = st.selectbox("Select Trading Pair", ["BTC/USDT", "ETH/USDT", "ADA/USDT"])
    
    # Time frame selector
    timeframe = st.selectbox("Time Frame", ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Generate sample OHLCV data
    dates = pd.date_range(start='2024-07-01', end='2024-07-24', freq='H')
    np.random.seed(42)
    
    # Create realistic price data
    base_price = 67000 if symbol == "BTC/USDT" else 3400
    price_changes = np.random.randn(len(dates)) * base_price * 0.01
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLCV data
    ohlcv_data = []
    for i, date in enumerate(dates):
        open_price = prices[i]
        high_price = open_price + abs(np.random.randn()) * open_price * 0.005
        low_price = open_price - abs(np.random.randn()) * open_price * 0.005
        close_price = open_price + np.random.randn() * open_price * 0.003
        volume = np.random.randint(100, 1000)
        
        ohlcv_data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(ohlcv_data)
    
    # Candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Technical Indicators")
        
        # Calculate simple moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Display current values
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("SMA 20", f"${sma_20:.2f}")
        st.metric("SMA 50", f"${sma_50:.2f}")
        
        # Signal
        if current_price > sma_20 > sma_50:
            st.success("🟢 Bullish Signal")
        elif current_price < sma_20 < sma_50:
            st.error("🔴 Bearish Signal")
        else:
            st.warning("🟡 Neutral Signal")
    
    with col2:
        st.subheader("Market Sentiment")
        
        # Sample sentiment data
        sentiment_data = {
            'Indicator': ['RSI', 'MACD', 'Bollinger Bands', 'Volume', 'Fear & Greed'],
            'Value': ['65.4', 'Bullish', 'Neutral', 'High', '72 (Greed)'],
            'Signal': ['Neutral', 'Buy', 'Hold', 'Positive', 'Caution']
        }
        
        sentiment_df = pd.DataFrame(sentiment_data)
        st.table(sentiment_df)

def show_portfolio():
    """Display portfolio information"""
    st.header("💼 Portfolio Management")
    
    # Portfolio overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Balance", "$12,345.67", "+$234.56")
    with col2:
        st.metric("Available Balance", "$2,345.67", "-$100.00")
    with col3:
        st.metric("In Positions", "$10,000.00", "+$334.56")
    
    # Holdings table
    st.subheader("Current Holdings")
    holdings_data = {
        'Asset': ['BTC', 'ETH', 'ADA', 'DOT', 'USDT'],
        'Amount': [0.15, 2.5, 1000, 50, 2345.67],
        'Value ($)': [10125.00, 8625.00, 450.00, 350.00, 2345.67],
        'Allocation (%)': [46.1, 39.3, 2.1, 1.6, 10.7],
        '24h Change (%)': [2.3, -1.2, 5.4, 0.8, 0.0]
    }
    
    holdings_df = pd.DataFrame(holdings_data)
    st.dataframe(holdings_df, use_container_width=True)
    
    # Position details
    st.subheader("Active Positions")
    positions_data = {
        'Pair': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        'Side': ['LONG', 'LONG', 'SHORT'],
        'Size': [0.1, 1.5, 500],
        'Entry Price': [66500, 3400, 0.48],
        'Current Price': [67500, 3450, 0.45],
        'P&L ($)': [100.00, 75.00, 15.00],
        'P&L (%)': [1.5, 1.47, 6.25]
    }
    
    positions_df = pd.DataFrame(positions_data)
    st.dataframe(positions_df, use_container_width=True)

def show_risk_management():
    """Display risk management settings and metrics"""
    st.header("⚠️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Parameters")
        
        max_daily_loss = st.slider("Max Daily Loss (%)", 1, 20, 5)
        max_position_risk = st.slider("Max Position Risk (%)", 1, 10, 3)
        max_correlation = st.slider("Max Correlation", 0.1, 1.0, 0.7)
        
        # Risk limits
        st.subheader("Current Risk Metrics")
        risk_metrics = {
            'Metric': ['Daily P&L', 'Portfolio Risk', 'Max Drawdown', 'Sharpe Ratio', 'VaR (95%)'],
            'Current': ['$234.56', '4.2%', '8.1%', '1.85', '$456.78'],
            'Limit': ['$617.28', '5.0%', '10.0%', '> 1.0', '$500.00'],
            'Status': ['✅ Safe', '✅ Safe', '✅ Safe', '✅ Good', '✅ Safe']
        }
        
        risk_df = pd.DataFrame(risk_metrics)
        st.table(risk_df)
    
    with col2:
        st.subheader("Risk Alerts")
        
        # Sample alerts
        alerts = [
            {"level": "info", "message": "Portfolio correlation within limits"},
            {"level": "warning", "message": "BTC position approaching size limit"},
            {"level": "success", "message": "Daily P&L target achieved"},
            {"level": "info", "message": "Risk metrics updated"}
        ]
        
        for alert in alerts:
            if alert["level"] == "warning":
                st.warning(f"⚠️ {alert['message']}")
            elif alert["level"] == "success":
                st.success(f"✅ {alert['message']}")
            else:
                st.info(f"ℹ️ {alert['message']}")
        
        # Risk chart
        st.subheader("Risk Over Time")
        dates = pd.date_range(start='2024-07-01', end='2024-07-24', freq='D')
        risk_values = np.random.uniform(2, 6, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=risk_values,
            mode='lines',
            name='Portfolio Risk (%)',
            line=dict(color='red', width=2)
        ))
        fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Risk Limit")
        fig.update_layout(
            title="Portfolio Risk Over Time",
            xaxis_title="Date",
            yaxis_title="Risk (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Display application settings"""
    st.header("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Exchange Settings", "Notifications", "General"])
    
    with tab1:
        st.subheader("Exchange Configuration")
        
        # Exchange selection
        exchanges = st.multiselect(
            "Select Exchanges",
            ["Binance", "Coinbase", "Kraken", "Bitfinex", "Huobi"],
            default=["Binance"]
        )
        
        # API credentials (placeholder)
        st.subheader("API Credentials")
        st.text_input("API Key", type="password", placeholder="Enter your API key")
        st.text_input("Secret Key", type="password", placeholder="Enter your secret key")
        st.text_input("Passphrase", type="password", placeholder="Enter passphrase (if required)")
        
        if st.button("Test Connection"):
            st.success("✅ Connection successful!")
    
    with tab2:
        st.subheader("Notification Settings")
        
        st.checkbox("Email Notifications", value=True)
        st.checkbox("Trade Alerts", value=True)
        st.checkbox("Risk Alerts", value=True)
        st.checkbox("Daily Reports", value=False)
        
        st.text_input("Email Address", placeholder="your@email.com")
        
    with tab3:
        st.subheader("General Settings")
        
        st.selectbox("Theme", ["Light", "Dark"], index=0)
        st.selectbox("Currency", ["USD", "EUR", "BTC"], index=0)
        st.selectbox("Timezone", ["UTC", "EST", "PST"], index=0)
        
        st.slider("Refresh Interval (seconds)", 1, 60, 5)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()


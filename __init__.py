"""
CryptoCaller - Advanced Cryptocurrency Trading Bot

A sophisticated cryptocurrency trading bot built with Streamlit, featuring
advanced technical analysis, risk management, and portfolio tracking capabilities.

Author: CryptoCaller Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "CryptoCaller Team"
__email__ = "support@cryptocaller.com"
__license__ = "MIT"

# Import main classes for easy access
from .market_data import MarketDataManager
from .trading_bot import CryptoTradingBot, TradingSignal, Position, Trade
from .portfolio_manager import PortfolioManager, Asset, Transaction
from .risk_management import RiskManager, RiskMetrics, RiskAlert

__all__ = [
    "MarketDataManager",
    "CryptoTradingBot",
    "TradingSignal",
    "Position",
    "Trade",
    "PortfolioManager",
    "Asset",
    "Transaction",
    "RiskManager",
    "RiskMetrics",
    "RiskAlert",
]


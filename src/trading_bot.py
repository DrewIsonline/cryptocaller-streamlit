import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, asdict

from src.trading_engine import TrendFollowingEngine
from src.market_data import MarketDataManager
from src.risk_management import RiskManager, RiskMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Portfolio:
    """Data class for portfolio information"""
    total_value: float = 0.0
    available_balance: float = 0.0
    positions: Dict = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

@dataclass
class Position:
    """Data class for trading positions"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass 
class Trade:
    """Data class for completed trades"""
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    trade_id: str = ""

@dataclass
class TradingConfig:
    """Data class for trading configuration"""
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1

@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    exchange: str
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    strategy: str
    entry_price: float
    stop_loss: float
    position_size: float
    reason: str
    timestamp: datetime

class CryptoTradingBot:
    """
    Main trading bot orchestrator that coordinates all trading activities
    Implements the FinRev-like functionality with trend-following strategies
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trading bot with configuration
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.is_running = False
        self.is_paused = False
        
        # Initialize components
        self.trading_engine = TrendFollowingEngine()
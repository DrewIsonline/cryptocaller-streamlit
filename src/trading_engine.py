"""
CryptoCaller Trading Engine
==========================

This module contains the core trading engines and strategies for the CryptoCaller application.
It implements various trading algorithms including trend following, mean reversion, and momentum strategies.

Author: Manus AI
Date: July 27, 2025
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import talib
import ccxt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    take_profit: float
    position_size: float
    reason: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'signal': self.signal,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }

class BaseTradingEngine(ABC):
    """Abstract base class for trading engines"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.signals_history = []
        self.performance_metrics = {}
    
    @abstractmethod
    async def generate_signal(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> float:
        """Calculate optimal position size"""
        pass
    
    def add_signal(self, signal: TradingSignal):
        """Add signal to history"""
        self.signals_history.append(signal)
        if len(self.signals_history) > 1000:  # Keep only last 1000 signals
            self.signals_history = self.signals_history[-1000:]
    
    def get_recent_signals(self, limit: int = 10) -> List[TradingSignal]:
        """Get recent signals"""
        return self.signals_history[-limit:]
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from signal history"""
        if not self.signals_history:
            return 0.0
        
        # This is a simplified calculation - in reality you'd track actual trade outcomes
        profitable_signals = sum(1 for signal in self.signals_history if signal.confidence > 0.6)
        return profitable_signals / len(self.signals_history) * 100

class TrendFollowingEngine(BaseTradingEngine):
    """
    Trend following trading engine using moving averages and momentum indicators
    """
    
    def __init__(self):
        super().__init__("TrendFollowing")
        self.short_ma_period = 20
        self.long_ma_period = 50
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.min_confidence = 0.5
    
    async def generate_signal(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trend following signal based on technical indicators
        
        Args:
            symbol: Trading pair symbol
            timeframe: Chart timeframe
            data: OHLCV data
            
        Returns:
            TradingSignal or None
        """
        try:
            if len(data) < self.long_ma_period + 10:
                self.logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Generate signal based on multiple conditions
            signal_data = self._analyze_trend(indicators, data)
            
            if signal_data['signal'] == 'hold':
                return None
            
            # Create trading signal
            current_price = float(data['close'].iloc[-1])
            
            signal = TradingSignal(
                symbol=symbol,
                exchange='binance',  # Default exchange
                signal=signal_data['signal'],
                confidence=signal_data['confidence'],
                strategy=self.name,
                entry_price=current_price,
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                position_size=0.0,  # Will be calculated separately
                reason=signal_data['reason'],
                timestamp=datetime.now()
            )
            
            self.add_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate technical indicators"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        indicators = {
            'sma_short': talib.SMA(close, timeperiod=self.short_ma_period),
            'sma_long': talib.SMA(close, timeperiod=self.long_ma_period),
            'ema_short': talib.EMA(close, timeperiod=self.short_ma_period),
            'ema_long': talib.EMA(close, timeperiod=self.long_ma_period),
            'rsi': talib.RSI(close, timeperiod=self.rsi_period),
            'macd': talib.MACD(close, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal),
            'bb_upper': talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0],
            'bb_middle': talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1],
            'bb_lower': talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2],
            'atr': talib.ATR(high, low, close, timeperiod=14),
            'adx': talib.ADX(high, low, close, timeperiod=14)
        }
        
        return indicators
    
    def _analyze_trend(self, indicators: Dict[str, np.ndarray], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend and generate signal"""
        current_price = float(data['close'].iloc[-1])
        
        # Get latest indicator values
        sma_short = indicators['sma_short'][-1]
        sma_long = indicators['sma_long'][-1]
        ema_short = indicators['ema_short'][-1]
        ema_long = indicators['ema_long'][-1]
        rsi = indicators['rsi'][-1]
        macd_line = indicators['macd'][0][-1]
        macd_signal = indicators['macd'][1][-1]
        macd_histogram = indicators['macd'][2][-1]
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        atr = indicators['atr'][-1]
        adx = indicators['adx'][-1]
        
        # Initialize signal components
        signal_strength = 0
        reasons = []
        
        # Moving Average Analysis
        if sma_short > sma_long and ema_short > ema_long:
            signal_strength += 2
            reasons.append("Bullish MA crossover")
        elif sma_short < sma_long and ema_short < ema_long:
            signal_strength -= 2
            reasons.append("Bearish MA crossover")
        
        # Price vs Moving Averages
        if current_price > sma_short > sma_long:
            signal_strength += 1
            reasons.append("Price above MAs")
        elif current_price < sma_short < sma_long:
            signal_strength -= 1
            reasons.append("Price below MAs")
        
        # RSI Analysis
        if rsi < 30:
            signal_strength += 1
            reasons.append("RSI oversold")
        elif rsi > 70:
            signal_strength -= 1
            reasons.append("RSI overbought")
        elif 40 < rsi < 60:
            signal_strength += 0.5
            reasons.append("RSI neutral")
        
        # MACD Analysis
        if macd_line > macd_signal and macd_histogram > 0:
            signal_strength += 1
            reasons.append("MACD bullish")
        elif macd_line < macd_signal and macd_histogram < 0:
            signal_strength -= 1
            reasons.append("MACD bearish")
        
        # ADX Trend Strength
        if adx > 25:
            signal_strength *= 1.2  # Amplify signal in strong trends
            reasons.append(f"Strong trend (ADX: {adx:.1f})")
        
        # Bollinger Bands
        if current_price < bb_lower:
            signal_strength += 0.5
            reasons.append("Price below BB lower")
        elif current_price > bb_upper:
            signal_strength -= 0.5
            reasons.append("Price above BB upper")
        
        # Determine final signal
        confidence = min(abs(signal_strength) / 5.0, 1.0)  # Normalize to 0-1
        
        if signal_strength >= 2 and confidence >= self.min_confidence:
            signal_type = 'buy'
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        elif signal_strength <= -2 and confidence >= self.min_confidence:
            signal_type = 'sell'
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        else:
            signal_type = 'hold'
            stop_loss = current_price
            take_profit = current_price
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': '; '.join(reasons)
        }
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> float:
        """
        Calculate position size using risk management principles
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade (0.01 = 1%)
            
        Returns:
            Position size in base currency
        """
        try:
            # Calculate risk amount
            risk_amount = account_balance * risk_per_trade
            
            # Calculate stop loss distance
            if signal.signal == 'buy':
                stop_distance = signal.entry_price - signal.stop_loss
            else:
                stop_distance = signal.stop_loss - signal.entry_price
            
            # Avoid division by zero
            if stop_distance <= 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / stop_distance
            
            # Apply confidence adjustment
            position_size *= signal.confidence
            
            # Ensure position size doesn't exceed maximum percentage of balance
            max_position_value = account_balance * 0.1  # Max 10% of balance per trade
            max_position_size = max_position_value / signal.entry_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

class MeanReversionEngine(BaseTradingEngine):
    """
    Mean reversion trading engine using statistical analysis
    """
    
    def __init__(self):
        super().__init__("MeanReversion")
        self.lookback_period = 20
        self.std_multiplier = 2.0
        self.rsi_period = 14
        self.min_confidence = 0.6
    
    async def generate_signal(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate mean reversion signal"""
        try:
            if len(data) < self.lookback_period + 10:
                return None
            
            close = data['close'].values
            current_price = close[-1]
            
            # Calculate statistical measures
            mean_price = np.mean(close[-self.lookback_period:])
            std_price = np.std(close[-self.lookback_period:])
            z_score = (current_price - mean_price) / std_price
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=self.rsi_period)[-1]
            
            # Generate signal
            signal_strength = 0
            reasons = []
            
            # Z-score analysis
            if z_score < -self.std_multiplier:
                signal_strength += 2
                reasons.append(f"Price {abs(z_score):.2f} std below mean")
            elif z_score > self.std_multiplier:
                signal_strength -= 2
                reasons.append(f"Price {z_score:.2f} std above mean")
            
            # RSI confirmation
            if rsi < 30 and z_score < -1:
                signal_strength += 1
                reasons.append("RSI oversold confirmation")
            elif rsi > 70 and z_score > 1:
                signal_strength -= 1
                reasons.append("RSI overbought confirmation")
            
            confidence = min(abs(signal_strength) / 3.0, 1.0)
            
            if signal_strength >= 2 and confidence >= self.min_confidence:
                signal_type = 'buy'
                stop_loss = current_price * 0.98
                take_profit = mean_price
            elif signal_strength <= -2 and confidence >= self.min_confidence:
                signal_type = 'sell'
                stop_loss = current_price * 1.02
                take_profit = mean_price
            else:
                return None
            
            signal = TradingSignal(
                symbol=symbol,
                exchange='binance',
                signal=signal_type,
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.0,
                reason='; '.join(reasons),
                timestamp=datetime.now()
            )
            
            self.add_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signal: {e}")
            return None
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> float:
        """Calculate position size for mean reversion strategy"""
        # Similar to trend following but with different risk parameters
        risk_amount = account_balance * risk_per_trade * 0.8  # Slightly more conservative
        
        if signal.signal == 'buy':
            stop_distance = signal.entry_price - signal.stop_loss
        else:
            stop_distance = signal.stop_loss - signal.entry_price
        
        if stop_distance <= 0:
            return 0.0
        
        position_size = risk_amount / stop_distance
        position_size *= signal.confidence
        
        max_position_value = account_balance * 0.08  # Max 8% for mean reversion
        max_position_size = max_position_value / signal.entry_price
        
        return min(position_size, max_position_size)

class MomentumEngine(BaseTradingEngine):
    """
    Momentum trading engine using price momentum and volume analysis
    """
    
    def __init__(self):
        super().__init__("Momentum")
        self.momentum_period = 10
        self.volume_period = 20
        self.min_confidence = 0.55
    
    async def generate_signal(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate momentum signal"""
        try:
            if len(data) < max(self.momentum_period, self.volume_period) + 10:
                return None
            
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            current_price = close[-1]
            
            # Calculate momentum indicators
            price_momentum = (current_price - close[-self.momentum_period]) / close[-self.momentum_period]
            volume_ratio = np.mean(volume[-5:]) / np.mean(volume[-self.volume_period:])
            
            # Calculate additional indicators
            rsi = talib.RSI(close, timeperiod=14)[-1]
            atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            signal_strength = 0
            reasons = []
            
            # Price momentum analysis
            if price_momentum > 0.02:  # 2% momentum
                signal_strength += 2
                reasons.append(f"Strong upward momentum: {price_momentum*100:.1f}%")
            elif price_momentum < -0.02:
                signal_strength -= 2
                reasons.append(f"Strong downward momentum: {price_momentum*100:.1f}%")
            
            # Volume confirmation
            if volume_ratio > 1.5:
                signal_strength += 1
                reasons.append("High volume confirmation")
            elif volume_ratio < 0.7:
                signal_strength -= 0.5
                reasons.append("Low volume warning")
            
            # RSI filter
            if 30 < rsi < 70:
                signal_strength *= 1.1  # Boost signal in neutral RSI range
                reasons.append("RSI in tradeable range")
            
            confidence = min(abs(signal_strength) / 3.0, 1.0)
            
            if signal_strength >= 2 and confidence >= self.min_confidence:
                signal_type = 'buy'
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 2.5)
            elif signal_strength <= -2 and confidence >= self.min_confidence:
                signal_type = 'sell'
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 2.5)
            else:
                return None
            
            signal = TradingSignal(
                symbol=symbol,
                exchange='binance',
                signal=signal_type,
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.0,
                reason='; '.join(reasons),
                timestamp=datetime.now()
            )
            
            self.add_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signal: {e}")
            return None
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float) -> float:
        """Calculate position size for momentum strategy"""
        # More aggressive position sizing for momentum
        risk_amount = account_balance * risk_per_trade * 1.2
        
        if signal.signal == 'buy':
            stop_distance = signal.entry_price - signal.stop_loss
        else:
            stop_distance = signal.stop_loss - signal.entry_price
        
        if stop_distance <= 0:
            return 0.0
        
        position_size = risk_amount / stop_distance
        position_size *= signal.confidence
        
        max_position_value = account_balance * 0.12  # Max 12% for momentum
        max_position_size = max_position_value / signal.entry_price
        
        return min(position_size, max_position_size)

class TradingEngineManager:
    """
    Manager class for coordinating multiple trading engines
    """
    
    def __init__(self):
        self.engines = {
            'trend_following': TrendFollowingEngine(),
            'mean_reversion': MeanReversionEngine(),
            'momentum': MomentumEngine()
        }
        self.active_engines = ['trend_following']  # Default active engine
        self.logger = logging.getLogger(__name__)
    
    def set_active_engines(self, engine_names: List[str]):
        """Set which engines are active"""
        valid_engines = [name for name in engine_names if name in self.engines]
        self.active_engines = valid_engines
        self.logger.info(f"Active engines set to: {valid_engines}")
    
    async def generate_signals(self, symbol: str, timeframe: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals from all active engines"""
        signals = []
        
        for engine_name in self.active_engines:
            engine = self.engines[engine_name]
            try:
                signal = await engine.generate_signal(symbol, timeframe, data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error in {engine_name} engine: {e}")
        
        return signals
    
    def get_consensus_signal(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Get consensus signal from multiple engines"""
        if not signals:
            return None
        
        # Simple consensus: majority vote weighted by confidence
        buy_weight = sum(s.confidence for s in signals if s.signal == 'buy')
        sell_weight = sum(s.confidence for s in signals if s.signal == 'sell')
        
        if buy_weight > sell_weight and buy_weight > 0.5:
            # Return the highest confidence buy signal
            buy_signals = [s for s in signals if s.signal == 'buy']
            return max(buy_signals, key=lambda x: x.confidence)
        elif sell_weight > buy_weight and sell_weight > 0.5:
            # Return the highest confidence sell signal
            sell_signals = [s for s in signals if s.signal == 'sell']
            return max(sell_signals, key=lambda x: x.confidence)
        
        return None
    
    def get_engine_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all engines"""
        performance = {}
        
        for name, engine in self.engines.items():
            performance[name] = {
                'total_signals': len(engine.signals_history),
                'win_rate': engine.calculate_win_rate(),
                'recent_signals': len(engine.get_recent_signals()),
                'avg_confidence': np.mean([s.confidence for s in engine.signals_history]) if engine.signals_history else 0
            }
        
        return performance

# Factory function for creating engines
def create_trading_engine(engine_type: str) -> BaseTradingEngine:
    """Factory function to create trading engines"""
    engines = {
        'trend_following': TrendFollowingEngine,
        'mean_reversion': MeanReversionEngine,
        'momentum': MomentumEngine
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    return engines[engine_type]()

# Export main classes
__all__ = [
    'TradingSignal',
    'BaseTradingEngine',
    'TrendFollowingEngine',
    'MeanReversionEngine',
    'MomentumEngine',
    'TradingEngineManager',
    'create_trading_engine'
]


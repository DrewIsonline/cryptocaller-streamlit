import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Data class for risk metrics"""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float  # Expected Shortfall (Conditional VaR)
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation_btc: float

@dataclass
class RiskAlert:
    """Data class for risk alerts"""
    timestamp: datetime
    level: str  # 'info', 'warning', 'critical'
    category: str  # 'position_size', 'correlation', 'drawdown', 'volatility'
    message: str
    current_value: float
    threshold: float
    symbol: Optional[str] = None

class RiskManager:
    """
    Comprehensive risk management system for cryptocurrency trading
    Monitors portfolio risk, position sizing, and generates alerts
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.alerts: List[RiskAlert] = []
        self.risk_history = []
        
        # Risk limits
        self.max_position_size = self.config.get('max_position_size', 10.0)  # % of portfolio
        self.max_daily_loss = self.config.get('max_daily_loss', 5.0)  # % of portfolio
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 15.0)  # %
        
        self._initialize_session_state()
    
    def _get_default_config(self) -> Dict:
        """Get default risk management configuration"""
        return {
            'max_position_size': 10.0,  # Maximum position size as % of portfolio
            'max_daily_loss': 5.0,      # Maximum daily loss as % of portfolio
            'max_correlation': 0.7,      # Maximum correlation between positions
            'max_drawdown_limit': 15.0,  # Maximum drawdown limit
            'var_confidence': 0.95,      # VaR confidence level
            'lookback_period': 30,       # Days for risk calculations
            'rebalance_threshold': 5.0,  # % deviation to trigger rebalancing
            'stop_loss_multiplier': 2.0, # Stop loss as multiple of daily volatility
        }
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'risk_alerts' not in st.session_state:
            st.session_state.risk_alerts = []
        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = None
    
    def calculate_portfolio_risk(self, portfolio_data: Dict, market_data: Dict) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        # Get portfolio returns for risk calculations
        returns = self._calculate_portfolio_returns(portfolio_data, market_data)
        
        if len(returns) < 10:  # Need minimum data for meaningful calculations
            return self._get_default_risk_metrics()
        
        # Calculate VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(returns, 1)  # 1st percentile for 99% VaR
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = np.mean(returns[returns <= var_95])
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Beta and correlation with BTC (simplified)
        btc_returns = self._get_btc_returns(market_data)
        if len(btc_returns) == len(returns):
            correlation_btc = np.corrcoef(returns, btc_returns)[0, 1]
            beta = np.cov(returns, btc_returns)[0, 1] / np.var(btc_returns) if np.var(btc_returns) > 0 else 1.0
        else:
            correlation_btc = 0.5  # Default assumption
            beta = 1.0
        
        risk_metrics = RiskMetrics(
            var_95=var_95 * 100,  # Convert to percentage
            var_99=var_99 * 100,
            expected_shortfall=expected_shortfall * 100,
            max_drawdown=abs(max_drawdown) * 100,
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            correlation_btc=correlation_btc
        )
        
        # Store in session state
        st.session_state.risk_metrics = risk_metrics
        
        return risk_metrics
    
    def _calculate_portfolio_returns(self, portfolio_data: Dict, market_data: Dict) -> List[float]:
        """Calculate historical portfolio returns"""
        # This is a simplified implementation
        # In practice, you would use actual historical price data
        
        # Generate synthetic returns based on current portfolio composition
        returns = []
        
        # Simulate 30 days of returns
        for _ in range(30):
            daily_return = 0.0
            
            # Weight returns by portfolio allocation
            for symbol, allocation in portfolio_data.get('allocations', {}).items():
                # Generate correlated returns for crypto assets
                asset_return = np.random.normal(0.001, 0.03)  # Mean 0.1%, std 3%
                daily_return += (allocation / 100) * asset_return
            
            returns.append(daily_return)
        
        return returns
    
    def _get_btc_returns(self, market_data: Dict) -> List[float]:
        """Get BTC returns for correlation calculation"""
        # Generate synthetic BTC returns
        return [np.random.normal(0.001, 0.04) for _ in range(30)]  # Higher volatility for BTC
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when insufficient data"""
        return RiskMetrics(
            var_95=2.5,
            var_99=4.0,
            expected_shortfall=3.0,
            max_drawdown=5.0,
            volatility=25.0,
            sharpe_ratio=1.2,
            beta=1.0,
            correlation_btc=0.6
        )
    
    def check_position_size_risk(self, symbol: str, position_value: float, portfolio_value: float) -> Optional[RiskAlert]:
        """Check if position size exceeds risk limits"""
        position_pct = (position_value / portfolio_value) * 100
        
        if position_pct > self.max_position_size:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level='warning',
                category='position_size',
                message=f"Position size for {symbol} ({position_pct:.1f}%) exceeds limit ({self.max_position_size}%)",
                current_value=position_pct,
                threshold=self.max_position_size,
                symbol=symbol
            )
            self._add_alert(alert)
            return alert
        
        return None
    
    def check_correlation_risk(self, portfolio_positions: Dict) -> List[RiskAlert]:
        """Check correlation risk between positions"""
        alerts = []
        
        # Simplified correlation check
        # In practice, you would calculate actual correlations using historical data
        
        crypto_positions = [symbol for symbol in portfolio_positions.keys() 
                          if symbol in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']]
        
        if len(crypto_positions) > 1:
            # Assume high correlation between crypto assets
            estimated_correlation = 0.8
            
            if estimated_correlation > self.max_correlation:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    level='warning',
                    category='correlation',
                    message=f"High correlation ({estimated_correlation:.2f}) detected between crypto positions",
                    current_value=estimated_correlation,
                    threshold=self.max_correlation
                )
                alerts.append(alert)
                self._add_alert(alert)
        
        return alerts
    
    def check_drawdown_risk(self, current_portfolio_value: float, peak_portfolio_value: float) -> Optional[RiskAlert]:
        """Check if current drawdown exceeds limits"""
        if peak_portfolio_value <= 0:
            return None
        
        drawdown_pct = ((peak_portfolio_value - current_portfolio_value) / peak_portfolio_value) * 100
        
        if drawdown_pct > self.max_drawdown_limit:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level='critical',
                category='drawdown',
                message=f"Portfolio drawdown ({drawdown_pct:.1f}%) exceeds limit ({self.max_drawdown_limit}%)",
                current_value=drawdown_pct,
                threshold=self.max_drawdown_limit
            )
            self._add_alert(alert)
            return alert
        
        return None
    
    def check_daily_loss_risk(self, daily_pnl: float, portfolio_value: float) -> Optional[RiskAlert]:
        """Check if daily loss exceeds limits"""
        daily_loss_pct = abs(daily_pnl / portfolio_value) * 100
        
        if daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level='critical',
                category='daily_loss',
                message=f"Daily loss ({daily_loss_pct:.1f}%) exceeds limit ({self.max_daily_loss}%)",
                current_value=daily_loss_pct,
                threshold=self.max_daily_loss
            )
            self._add_alert(alert)
            return alert
        
        return None
    
    def calculate_position_size(self, signal_confidence: float, portfolio_value: float, 
                              volatility: float, stop_loss_distance: float) -> float:
        """Calculate optimal position size using Kelly Criterion and risk management"""
        
        # Base position size as percentage of portfolio
        base_size_pct = self.config.get('base_position_size', 5.0)
        
        # Adjust based on signal confidence
        confidence_multiplier = signal_confidence  # 0.0 to 1.0
        
        # Adjust based on volatility (lower size for higher volatility)
        volatility_adjustment = max(0.5, 1.0 - (volatility / 50.0))  # Assume 50% is max volatility
        
        # Calculate position size
        position_size_pct = base_size_pct * confidence_multiplier * volatility_adjustment
        
        # Apply maximum position size limit
        position_size_pct = min(position_size_pct, self.max_position_size)
        
        # Convert to dollar amount
        position_size = (position_size_pct / 100) * portfolio_value
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, volatility: float, side: str = 'long') -> float:
        """Calculate stop loss based on volatility"""
        
        # Use multiple of daily volatility for stop loss
        stop_distance = volatility * self.config.get('stop_loss_multiplier', 2.0)
        
        if side == 'long':
            stop_loss = entry_price * (1 - stop_distance / 100)
        else:  # short
            stop_loss = entry_price * (1 + stop_distance / 100)
        
        return stop_loss
    
    def _add_alert(self, alert: RiskAlert):
        """Add alert to the system"""
        self.alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
        
        # Update session state
        st.session_state.risk_alerts = self.alerts[-10:]  # Show last 10 in UI
        
        logger.warning(f"Risk Alert: {alert.message}")
    
    def get_recent_alerts(self, limit: int = 10) -> List[RiskAlert]:
        """Get recent risk alerts"""
        return sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_risk_summary(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """Get comprehensive risk summary"""
        
        # Calculate current risk metrics
        risk_metrics = self.calculate_portfolio_risk(portfolio_data, market_data)
        
        # Count alerts by level
        recent_alerts = self.get_recent_alerts(24)  # Last 24 alerts
        alert_counts = {
            'critical': len([a for a in recent_alerts if a.level == 'critical']),
            'warning': len([a for a in recent_alerts if a.level == 'warning']),
            'info': len([a for a in recent_alerts if a.level == 'info'])
        }
        
        # Risk score (0-100, lower is better)
        risk_score = self._calculate_risk_score(risk_metrics, alert_counts)
        
        return {
            'risk_metrics': risk_metrics,
            'risk_score': risk_score,
            'alert_counts': alert_counts,
            'recent_alerts': recent_alerts[:5],  # Top 5 most recent
            'risk_level': self._get_risk_level(risk_score)
        }
    
    def _calculate_risk_score(self, risk_metrics: RiskMetrics, alert_counts: Dict) -> float:
        """Calculate overall risk score (0-100)"""
        
        # Base score from risk metrics
        score = 0
        
        # VaR component (0-30 points)
        var_score = min(30, abs(risk_metrics.var_95) * 3)
        score += var_score
        
        # Volatility component (0-25 points)
        vol_score = min(25, risk_metrics.volatility)
        score += vol_score
        
        # Drawdown component (0-25 points)
        dd_score = min(25, risk_metrics.max_drawdown * 1.5)
        score += dd_score
        
        # Alert component (0-20 points)
        alert_score = min(20, alert_counts['critical'] * 10 + alert_counts['warning'] * 3)
        score += alert_score
        
        return min(100, score)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level description"""
        if risk_score < 30:
            return "Low"
        elif risk_score < 60:
            return "Medium"
        elif risk_score < 80:
            return "High"
        else:
            return "Critical"
    
    def generate_risk_report(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """Generate comprehensive risk report"""
        
        risk_summary = self.get_risk_summary(portfolio_data, market_data)
        
        # Risk recommendations
        recommendations = []
        
        if risk_summary['risk_score'] > 70:
            recommendations.append("Consider reducing position sizes")
            recommendations.append("Review stop-loss levels")
        
        if risk_summary['risk_metrics'].correlation_btc > 0.8:
            recommendations.append("Diversify beyond crypto-correlated assets")
        
        if risk_summary['risk_metrics'].max_drawdown > 10:
            recommendations.append("Implement stricter drawdown controls")
        
        if risk_summary['alert_counts']['critical'] > 0:
            recommendations.append("Address critical risk alerts immediately")
        
        return {
            'summary': risk_summary,
            'recommendations': recommendations,
            'timestamp': datetime.now(),
            'config': self.config
        }
    
    def update_risk_limits(self, new_limits: Dict):
        """Update risk management limits"""
        self.config.update(new_limits)
        
        # Update instance variables
        self.max_position_size = self.config.get('max_position_size', 10.0)
        self.max_daily_loss = self.config.get('max_daily_loss', 5.0)
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 15.0)
        
        logger.info("Risk limits updated")
    
    def get_position_risk_analysis(self, symbol: str, position_size: float, 
                                 entry_price: float, current_price: float) -> Dict:
        """Analyze risk for a specific position"""
        
        # Calculate position metrics
        position_value = position_size * current_price
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_pct = (unrealized_pnl / (entry_price * position_size)) * 100
        
        # Estimate position volatility (simplified)
        estimated_volatility = 30.0  # Default 30% annual volatility for crypto
        
        # Calculate VaR for this position
        daily_var = position_value * (estimated_volatility / 100) / np.sqrt(252) * 1.65  # 95% confidence
        
        return {
            'symbol': symbol,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'estimated_volatility': estimated_volatility,
            'daily_var_95': daily_var,
            'risk_level': 'High' if abs(unrealized_pnl_pct) > 10 else 'Medium' if abs(unrealized_pnl_pct) > 5 else 'Low'
        }


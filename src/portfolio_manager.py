import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Asset:
    """Data class for portfolio assets"""
    symbol: str
    amount: float
    current_price: float
    avg_cost: float
    total_value: float
    unrealized_pnl: float
    allocation_pct: float

@dataclass
class Transaction:
    """Data class for portfolio transactions"""
    timestamp: datetime
    symbol: str
    transaction_type: str  # 'buy', 'sell', 'deposit', 'withdrawal'
    amount: float
    price: float
    value: float
    fee: float
    notes: str

class PortfolioManager:
    """
    Manages portfolio tracking, asset allocation, and performance analysis
    Adapted for Streamlit interface with session state management
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.assets: Dict[str, Asset] = {}
        self.transactions: List[Transaction] = []
        self.cash_balance = initial_balance
        
        # Performance tracking
        self.portfolio_history = []
        self.daily_returns = []
        
        # Initialize with some demo data
        self._initialize_demo_portfolio()
        self._initialize_session_state()
    
    def _initialize_demo_portfolio(self):
        """Initialize portfolio with demo data"""
        # Add some initial holdings for demonstration
        demo_assets = {
            'BTC': {'amount': 0.15, 'avg_cost': 65000},
            'ETH': {'amount': 2.5, 'avg_cost': 3300},
            'ADA': {'amount': 1000, 'avg_cost': 0.42},
            'DOT': {'amount': 50, 'avg_cost': 6.8},
        }
        
        for symbol, data in demo_assets.items():
            # Simulate current prices with some variation
            current_price = data['avg_cost'] * np.random.uniform(0.95, 1.15)
            amount = data['amount']
            total_value = amount * current_price
            unrealized_pnl = (current_price - data['avg_cost']) * amount
            
            asset = Asset(
                symbol=symbol,
                amount=amount,
                current_price=current_price,
                avg_cost=data['avg_cost'],
                total_value=total_value,
                unrealized_pnl=unrealized_pnl,
                allocation_pct=0.0  # Will be calculated later
            )
            
            self.assets[symbol] = asset
            
            # Add initial transaction
            transaction = Transaction(
                timestamp=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                symbol=symbol,
                transaction_type='buy',
                amount=amount,
                price=data['avg_cost'],
                value=amount * data['avg_cost'],
                fee=amount * data['avg_cost'] * 0.001,  # 0.1% fee
                notes='Initial purchase'
            )
            self.transactions.append(transaction)
        
        # Update cash balance
        total_invested = sum(asset.avg_cost * asset.amount for asset in self.assets.values())
        self.cash_balance = max(0, self.initial_balance - total_invested)
        
        # Calculate allocations
        self._update_allocations()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'portfolio_assets' not in st.session_state:
            st.session_state.portfolio_assets = self.assets
        if 'portfolio_transactions' not in st.session_state:
            st.session_state.portfolio_transactions = self.transactions
        if 'portfolio_cash_balance' not in st.session_state:
            st.session_state.portfolio_cash_balance = self.cash_balance
    
    def update_asset_prices(self, price_data: Dict[str, float]):
        """Update current prices for all assets"""
        for symbol, asset in self.assets.items():
            if symbol in price_data or f"{symbol}/USDT" in price_data:
                # Handle both symbol formats
                price_key = symbol if symbol in price_data else f"{symbol}/USDT"
                new_price = price_data[price_key]
                
                asset.current_price = new_price
                asset.total_value = asset.amount * new_price
                asset.unrealized_pnl = (new_price - asset.avg_cost) * asset.amount
        
        self._update_allocations()
        st.session_state.portfolio_assets = self.assets
    
    def _update_allocations(self):
        """Update allocation percentages for all assets"""
        total_portfolio_value = self.get_total_portfolio_value()
        
        for asset in self.assets.values():
            asset.allocation_pct = (asset.total_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
    
    def add_transaction(self, symbol: str, transaction_type: str, amount: float, price: float, fee: float = 0.0, notes: str = ""):
        """Add a new transaction to the portfolio"""
        value = amount * price
        
        transaction = Transaction(
            timestamp=datetime.now(),
            symbol=symbol,
            transaction_type=transaction_type,
            amount=amount,
            price=price,
            value=value,
            fee=fee,
            notes=notes
        )
        
        self.transactions.append(transaction)
        
        # Update asset holdings
        if transaction_type == 'buy':
            self._add_to_position(symbol, amount, price)
            self.cash_balance -= (value + fee)
        elif transaction_type == 'sell':
            self._reduce_position(symbol, amount, price)
            self.cash_balance += (value - fee)
        
        self._update_allocations()
        st.session_state.portfolio_transactions = self.transactions[-50:]  # Keep last 50 transactions
        st.session_state.portfolio_cash_balance = self.cash_balance
    
    def _add_to_position(self, symbol: str, amount: float, price: float):
        """Add to existing position or create new one"""
        if symbol in self.assets:
            # Update existing position with weighted average cost
            existing_asset = self.assets[symbol]
            total_amount = existing_asset.amount + amount
            total_cost = (existing_asset.avg_cost * existing_asset.amount) + (price * amount)
            new_avg_cost = total_cost / total_amount
            
            existing_asset.amount = total_amount
            existing_asset.avg_cost = new_avg_cost
            existing_asset.total_value = total_amount * existing_asset.current_price
            existing_asset.unrealized_pnl = (existing_asset.current_price - new_avg_cost) * total_amount
        else:
            # Create new position
            asset = Asset(
                symbol=symbol,
                amount=amount,
                current_price=price,
                avg_cost=price,
                total_value=amount * price,
                unrealized_pnl=0.0,
                allocation_pct=0.0
            )
            self.assets[symbol] = asset
    
    def _reduce_position(self, symbol: str, amount: float, price: float):
        """Reduce existing position"""
        if symbol in self.assets:
            existing_asset = self.assets[symbol]
            
            if amount >= existing_asset.amount:
                # Close entire position
                del self.assets[symbol]
            else:
                # Reduce position
                existing_asset.amount -= amount
                existing_asset.total_value = existing_asset.amount * existing_asset.current_price
                existing_asset.unrealized_pnl = (existing_asset.current_price - existing_asset.avg_cost) * existing_asset.amount
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value including cash"""
        asset_value = sum(asset.total_value for asset in self.assets.values())
        return asset_value + self.cash_balance
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_value = self.get_total_portfolio_value()
        total_unrealized_pnl = sum(asset.unrealized_pnl for asset in self.assets.values())
        total_invested = sum(asset.avg_cost * asset.amount for asset in self.assets.values())
        
        # Calculate returns
        total_return = total_value - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return {
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'asset_value': total_value - self.cash_balance,
            'total_invested': total_invested,
            'unrealized_pnl': total_unrealized_pnl,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_assets': len(self.assets),
            'largest_holding': max(self.assets.values(), key=lambda x: x.allocation_pct).symbol if self.assets else None
        }
    
    def get_asset_allocation(self) -> pd.DataFrame:
        """Get asset allocation as DataFrame"""
        if not self.assets:
            return pd.DataFrame()
        
        data = []
        for asset in self.assets.values():
            data.append({
                'Symbol': asset.symbol,
                'Amount': asset.amount,
                'Current Price': asset.current_price,
                'Total Value': asset.total_value,
                'Allocation %': asset.allocation_pct,
                'Unrealized P&L': asset.unrealized_pnl,
                'Avg Cost': asset.avg_cost
            })
        
        return pd.DataFrame(data)
    
    def get_transaction_history(self, limit: int = 50) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        if not self.transactions:
            return pd.DataFrame()
        
        # Sort by timestamp (most recent first)
        sorted_transactions = sorted(self.transactions, key=lambda x: x.timestamp, reverse=True)
        limited_transactions = sorted_transactions[:limit]
        
        data = []
        for tx in limited_transactions:
            data.append({
                'Date': tx.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Symbol': tx.symbol,
                'Type': tx.transaction_type.upper(),
                'Amount': tx.amount,
                'Price': tx.price,
                'Value': tx.value,
                'Fee': tx.fee,
                'Notes': tx.notes
            })
        
        return pd.DataFrame(data)
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        if len(self.portfolio_history) < 2:
            return {
                'daily_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.portfolio_history)):
            daily_return = (self.portfolio_history[i] - self.portfolio_history[i-1]) / self.portfolio_history[i-1]
            returns.append(daily_return)
        
        if not returns:
            return {
                'daily_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        # Calculate metrics
        avg_daily_return = np.mean(returns)
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (avg_daily_return * 252) / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = self.portfolio_history[0]
        max_drawdown = 0
        for value in self.portfolio_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate (days with positive returns)
        positive_days = sum(1 for r in returns if r > 0)
        win_rate = (positive_days / len(returns)) * 100 if returns else 0
        
        return {
            'daily_return': avg_daily_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate
        }
    
    def update_portfolio_history(self):
        """Update portfolio value history for performance tracking"""
        current_value = self.get_total_portfolio_value()
        self.portfolio_history.append(current_value)
        
        # Keep only last 365 days
        if len(self.portfolio_history) > 365:
            self.portfolio_history = self.portfolio_history[-365:]
    
    def get_portfolio_chart_data(self) -> pd.DataFrame:
        """Get portfolio value history for charting"""
        if not self.portfolio_history:
            # Generate sample data for demo
            dates = pd.date_range(start='2024-01-01', end='2024-07-24', freq='D')
            values = []
            current_value = self.initial_balance
            
            for _ in dates:
                daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
                current_value *= (1 + daily_change)
                values.append(current_value)
            
            return pd.DataFrame({
                'Date': dates,
                'Portfolio Value': values
            })
        
        # Use actual history if available
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=len(self.portfolio_history)-1),
            periods=len(self.portfolio_history),
            freq='D'
        )
        
        return pd.DataFrame({
            'Date': dates,
            'Portfolio Value': self.portfolio_history
        })
    
    def rebalance_portfolio(self, target_allocations: Dict[str, float]) -> List[Dict]:
        """
        Calculate rebalancing trades to achieve target allocations
        Returns list of recommended trades
        """
        total_value = self.get_total_portfolio_value()
        current_allocations = {asset.symbol: asset.allocation_pct for asset in self.assets.values()}
        
        trades = []
        
        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0)
            difference_pct = target_pct - current_pct
            
            if abs(difference_pct) > 1:  # Only rebalance if difference > 1%
                target_value = (target_pct / 100) * total_value
                current_value = self.assets[symbol].total_value if symbol in self.assets else 0
                trade_value = target_value - current_value
                
                if trade_value > 0:
                    # Need to buy
                    trades.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'value': trade_value,
                        'current_allocation': current_pct,
                        'target_allocation': target_pct
                    })
                else:
                    # Need to sell
                    trades.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'value': abs(trade_value),
                        'current_allocation': current_pct,
                        'target_allocation': target_pct
                    })
        
        return trades
    
    def get_diversification_metrics(self) -> Dict:
        """Calculate portfolio diversification metrics"""
        if not self.assets:
            return {
                'concentration_ratio': 0,
                'herfindahl_index': 0,
                'effective_assets': 0
            }
        
        allocations = [asset.allocation_pct / 100 for asset in self.assets.values()]
        
        # Concentration ratio (top 3 holdings)
        sorted_allocations = sorted(allocations, reverse=True)
        concentration_ratio = sum(sorted_allocations[:3])
        
        # Herfindahl-Hirschman Index
        herfindahl_index = sum(allocation ** 2 for allocation in allocations)
        
        # Effective number of assets
        effective_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'concentration_ratio': concentration_ratio * 100,
            'herfindahl_index': herfindahl_index,
            'effective_assets': effective_assets
        }


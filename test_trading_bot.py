import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_bot import CryptoTradingBot

class TestCryptoTradingBot(unittest.TestCase):
    def setUp(self):
        self.bot = CryptoTradingBot()
    
    def test_bot_initialization(self):
        self.assertIsNotNone(self.bot)
        self.assertEqual(self.bot.status, 'stopped')
    
    @patch('trading_bot.MarketDataManager')
    def test_market_data_connection(self, mock_market_data):
        mock_market_data.return_value.get_price.return_value = 50000
        price = self.bot.get_current_price('BTC/USDT')
        self.assertEqual(price, 50000)
    
    def test_risk_validation(self):
        # Test risk management validation
        result = self.bot.validate_trade_risk('BTC/USDT', 1000, 0.1)
        self.assertIsInstance(result, bool)

if __name__ == '__main__':
    unittest.main()
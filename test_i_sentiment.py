# test_integration.py (CORRECTED)

import os
import sys
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytz

# --- Setup Python path for imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'afa_core')))

# --- Import REAL classes ---
from sentiment_analyzer import SentimentAnalyzer
from data_handler import DataHandler, MarketDataClient
from signal_generator import SignalGenerator

# Configure logging
import logging
logging.basicConfig(level=logging.CRITICAL)


# ✅ --- FIX: Move decorators from setUp to the class level ---
@patch.dict(os.environ, {"APCA_API_KEY_ID": "mock_key", "APCA_API_SECRET_KEY": "mock_secret"})
@patch('data_handler.TradingClient')
@patch('data_handler.StockHistoricalDataClient')
@patch('yfinance.Ticker')
@patch('sentiment_analyzer.requests.post')
@patch('sentiment_analyzer.requests.get')
class TestIntegration(unittest.TestCase):

    # The mock objects are now passed to EACH test method, not setUp
    def setUp(self):
        # We don't need the mock arguments here anymore
        # Initialize the components
        self.sentiment_analyzer = SentimentAnalyzer(llama_model="llama3.2:1b")
        self.data_handler = DataHandler()
        self.signal_generator = None # Will be initialized later with fetched data

    def _create_mock_ohlcv_data(self, symbol="MOCKCO", days=20):
        """Helper to create a mock OHLCV DataFrame."""
        end_date = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D', tz=pytz.utc)
        dates = dates[dates.weekday < 5] # Filter weekends

        data = {
            'open': np.linspace(100, 105, len(dates)),
            'high': np.linspace(101, 106, len(dates)),
            'low': np.linspace(99, 104, len(dates)),
            'close': np.linspace(100.5, 105.5, len(dates)),
            'volume': np.random.randint(100000, 500000, len(dates)),
            'trade_count': np.random.randint(1000, 5000, len(dates)),
            'vwap': np.linspace(100.2, 105.2, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'timestamp'
        
        multi_index = pd.MultiIndex.from_product([[symbol], df.index], names=['symbol', 'timestamp'])
        df_multi = df.set_index(multi_index)
        df_multi.columns = df_multi.columns.str.lower()
        
        return df_multi

    def _create_mock_news(self, symbol: str, sentiment: str, days_back: int = 7) -> list:
        """Helper to create mock news items."""
        now_ts = int(datetime.now().timestamp())
        news_items = []

        if sentiment == "Positive":
            news_items.extend([
                {"title": f"BREAKING: {symbol} announces record profits!", "link": "http://example.com/pos_news1", "providerPublishTime": now_ts - 3600*24*1},
                {"title": f"{symbol} stock upgraded to 'Strong Buy'.", "link": "http://example.com/pos_news2", "providerPublishTime": now_ts - 3600*24*2},
            ])
        elif sentiment == "Negative":
            news_items.extend([
                {"title": f"WARNING: {symbol} shares tumble.", "link": "http://example.com/neg_news1", "providerPublishTime": now_ts - 3600*24*1},
                {"title": f"Analysts downgrade {symbol}.", "link": "http://example.com/neg_news2", "providerPublishTime": now_ts - 3600*24*2},
            ])
        else: # Neutral
            news_items.extend([
                {"title": f"UPDATE: {symbol} shows stable performance.", "link": "http://example.com/neu_news1", "providerPublishTime": now_ts - 3600*24*1},
                {"title": f"{symbol} shares unchanged.", "link": "http://example.com/neu_news2", "providerPublishTime": now_ts - 3600*24*2},
            ])
        
        news_items.append({"title": f"OLD NEWS.", "link": "http://example.com/old_news", "providerPublishTime": now_ts - 3600*24*(days_back + 1)})
        return news_items

    # ✅ --- FIX: Add the mock arguments to each test method's signature ---
    # The order of arguments must match the order of decorators (from bottom up)
    def test_end_to_end_integration_positive_scenario(self, mock_requests_get, mock_requests_post, mock_yfinance_ticker, mock_historical_client, mock_trading_client):
        """Tests the full pipeline for a BUY signal."""
        symbol = "AAPL"
        
        # --- Configure mocks for THIS test ---
        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.json.return_value = {'models': [{'name': 'llama3.2:1b'}]}
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'longName': 'Apple Inc.'}
        mock_ticker_instance.news = self._create_mock_news(symbol, "Positive")
        mock_yfinance_ticker.return_value = mock_ticker_instance
        # ---

        # Step 1: DataHandler fetches market data
        start_date = datetime.now(pytz.utc) - timedelta(days=30)
        end_date = datetime.now(pytz.utc)
        
        self.mock_ohlcv_data = self._create_mock_ohlcv_data(symbol, days=20)
        last_timestamp = self.mock_ohlcv_data.index.get_level_values('timestamp').max()
        self.mock_ohlcv_data.loc[(symbol, last_timestamp), 'close'] = 160.0

        mock_historical_client.return_value.get_stock_bars.return_value.df = self.mock_ohlcv_data
        
        market_data_df = self.data_handler.fetch_historical_stock_data(symbols=symbol, start_date=start_date, end_date=end_date, flatten=True)
        
        self.signal_generator = SignalGenerator(df_ohlcv=market_data_df)
        self.signal_generator.add_composite_signal()

        # Step 2: SentimentAnalyzer analyzes news
        mock_requests_post.return_value.status_code = 200
        mock_requests_post.return_value.json.return_value = {'response': 'POSITIVE'}
            
        news_sentiment = self.sentiment_analyzer.get_news_sentiment_for_symbol(symbol, days_back=7)
        
        self.assertEqual(news_sentiment.get('overall_sentiment'), 'Positive')
        self.assertGreater(news_sentiment.get('total_articles'), 0)

        # Step 3: Combined decision logic
        ta_signals = self.signal_generator.get_current_signals()
        final_action = "HOLD"
        if news_sentiment.get('overall_sentiment') == "Positive" and ta_signals.get('composite_action', 0) >= 0:
            final_action = "BUY"
        
        self.assertEqual(final_action, 'BUY')
        print(f"\n✅ Positive Scenario Test Passed: Final Action = {final_action}")

    def test_end_to_end_integration_negative_scenario(self, mock_requests_get, mock_requests_post, mock_yfinance_ticker, mock_historical_client, mock_trading_client):
        """Tests the full pipeline for a SELL signal."""
        symbol = "MSFT" 

        # --- Configure mocks for THIS test ---
        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.json.return_value = {'models': [{'name': 'llama3.2:1b'}]}
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'longName': 'Microsoft Corp.'}
        mock_ticker_instance.news = self._create_mock_news(symbol, "Negative")
        mock_yfinance_ticker.return_value = mock_ticker_instance
        # ---

        # Step 1: DataHandler fetches market data
        start_date = datetime.now(pytz.utc) - timedelta(days=30)
        end_date = datetime.now(pytz.utc)
        
        self.mock_ohlcv_data = self._create_mock_ohlcv_data(symbol, days=20)
        last_timestamp = self.mock_ohlcv_data.index.get_level_values('timestamp').max()
        self.mock_ohlcv_data.loc[(symbol, last_timestamp), 'close'] = 140.0

        mock_historical_client.return_value.get_stock_bars.return_value.df = self.mock_ohlcv_data
        
        market_data_df = self.data_handler.fetch_historical_stock_data(symbols=symbol, start_date=start_date, end_date=end_date, flatten=True)
        
        self.signal_generator = SignalGenerator(df_ohlcv=market_data_df)
        self.signal_generator.add_composite_signal()

        # Step 2: SentimentAnalyzer analyzes news
        mock_requests_post.return_value.status_code = 200
        mock_requests_post.return_value.json.return_value = {'response': 'NEGATIVE'}

        news_sentiment = self.sentiment_analyzer.get_news_sentiment_for_symbol(symbol, days_back=7)
        
        self.assertEqual(news_sentiment.get('overall_sentiment'), 'Negative')
        self.assertGreater(news_sentiment.get('total_articles'), 0)

        # Step 3: Combined decision logic
        ta_signals = self.signal_generator.get_current_signals()
        final_action = "HOLD"
        if news_sentiment.get('overall_sentiment') == "Negative" and ta_signals.get('composite_action', 0) <= 0:
            final_action = "SELL"
        
        self.assertEqual(final_action, 'SELL')
        print(f"✅ Negative Scenario Test Passed: Final Action = {final_action}")

    def test_end_to_end_integration_neutral_scenario(self, mock_requests_get, mock_requests_post, mock_yfinance_ticker, mock_historical_client, mock_trading_client):
        """Tests the full pipeline for a HOLD signal."""
        symbol = "GOOG"

        # --- Configure mocks for THIS test ---
        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.json.return_value = {'models': [{'name': 'llama3.2:1b'}]}
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'longName': 'Alphabet Inc.'}
        mock_ticker_instance.news = self._create_mock_news(symbol, "Neutral")
        mock_yfinance_ticker.return_value = mock_ticker_instance
        # ---

        # Step 1: DataHandler fetches market data
        start_date = datetime.now(pytz.utc) - timedelta(days=30)
        end_date = datetime.now(pytz.utc)
        
        self.mock_ohlcv_data = self._create_mock_ohlcv_data(symbol, days=20)
        last_timestamp = self.mock_ohlcv_data.index.get_level_values('timestamp').max()
        self.mock_ohlcv_data.loc[(symbol, last_timestamp), 'close'] = 150.0

        mock_historical_client.return_value.get_stock_bars.return_value.df = self.mock_ohlcv_data
        
        market_data_df = self.data_handler.fetch_historical_stock_data(symbols=symbol, start_date=start_date, end_date=end_date, flatten=True)
        
        self.signal_generator = SignalGenerator(df_ohlcv=market_data_df)
        self.signal_generator.add_composite_signal()

        # Step 2: SentimentAnalyzer analyzes news
        mock_requests_post.return_value.status_code = 200
        mock_requests_post.return_value.json.return_value = {'response': 'NEUTRAL'}

        news_sentiment = self.sentiment_analyzer.get_news_sentiment_for_symbol(symbol, days_back=7)
        
        self.assertEqual(news_sentiment.get('overall_sentiment'), 'Neutral')
        self.assertGreater(news_sentiment.get('total_articles'), 0)

        # Step 3: Combined decision logic
        ta_signals = self.signal_generator.get_current_signals()
        final_action = "HOLD" # Default
        if news_sentiment.get('overall_sentiment') == "Positive":
            final_action = "BUY"
        elif news_sentiment.get('overall_sentiment') == "Negative":
            final_action = "SELL"
        
        self.assertEqual(final_action, 'HOLD')
        print(f"✅ Neutral Scenario Test Passed: Final Action = {final_action}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)
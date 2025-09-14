import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'afa_core')))

from data_handler import DataHandler, MarketDataClient, BrokerageClient

from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide

import logging
logging.basicConfig(level=logging.CRITICAL)


class TestDataHandler(unittest.TestCase):

    @patch('data_handler.MarketDataClient')
    @patch('data_handler.BrokerageClient')
    @patch.dict(os.environ, {"APCA_API_KEY_ID": "mock_key", "APCA_API_SECRET_KEY": "mock_secret"})
    def setUp(self, MockBrokerageClient, MockMarketDataClient):
        self.data_handler = DataHandler()

        self.mock_market_data = self.data_handler.market_data
        self.mock_brokerage = self.data_handler.brokerage

        self.mock_market_data.fetch_historical_stock_data.return_value = pd.DataFrame() 

        self.mock_account = MagicMock()
        self.mock_account.status = 'ACTIVE'
        self.mock_account.buying_power = '200000'
        self.mock_account.portfolio_value = '100000'
        self.mock_brokerage.get_account_details.return_value = self.mock_account

        self.mock_brokerage.get_all_positions.return_value = [] 
        
        self.mock_order_response = MagicMock(client_order_id='mock_order_id_123', status='accepted')
        self.mock_brokerage.submit_market_order.return_value = self.mock_order_response


    def _create_mock_alpaca_bars_df(self, symbol="TEST", num_days=5, is_multi_index=True):
        """Helper to create a mock Alpaca-like OHLCV DataFrame."""
        end_date = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        dates = pd.date_range(end=end_date, periods=num_days, freq='D', tz=pytz.utc)
        
        data = {
            'open': np.random.rand(num_days) * 10 + 100,
            'high': np.random.rand(num_days) * 10 + 105,
            'low': np.random.rand(num_days) * 10 + 95,
            'close': np.random.rand(num_days) * 10 + 100,
            'volume': np.random.randint(10000, 100000, num_days),
            'trade_count': np.random.randint(100, 1000, num_days),
            'vwap': np.random.rand(num_days) * 10 + 100,
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'timestamp'
        
        if is_multi_index:
            multi_index = pd.MultiIndex.from_product([[symbol], df.index], names=['symbol', 'timestamp'])
            df = df.set_index(multi_index)
        
        df.columns = df.columns.str.lower()
        return df


    # --- Tests for fetch_historical_stock_data ---

    def test_fetch_historical_stock_data_success_single_symbol(self):
        """
        Test successful fetching of historical data for a single symbol.
        Expects the MarketDataClient to handle flattening when requested.
        """
        symbol = "AAPL"
        mock_flat_df = self._create_mock_alpaca_bars_df(symbol=symbol, num_days=10, is_multi_index=False)
        self.mock_market_data.fetch_historical_stock_data.return_value = mock_flat_df

        start_date = datetime.now() - timedelta(days=10)
        end_date = datetime.now()
        
        df = self.data_handler.fetch_historical_stock_data(
            symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            flatten=True # Explicitly pass flatten=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(df.index.nlevels, 1)
        self.assertTrue(all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.mock_market_data.fetch_historical_stock_data.assert_called_once()
        # DataHandler explicitly passed flatten=True, so assertion should include it.
        # Still no 'timeframe' because DataHandler doesn't explicitly pass it.
        self.mock_market_data.fetch_historical_stock_data.assert_called_once_with(
            symbols=symbol, start_date=start_date, end_date=end_date, flatten=True
        )


    def test_fetch_historical_stock_data_success_multiple_symbols(self):
        """Test successful fetching of historical data for multiple symbols."""
        symbols = ["AAPL", "MSFT"]
        mock_df_aapl = self._create_mock_alpaca_bars_df(symbol="AAPL", num_days=5, is_multi_index=True)
        mock_df_msft = self._create_mock_alpaca_bars_df(symbol="MSFT", num_days=5, is_multi_index=True)
        mock_combined_df = pd.concat([mock_df_aapl, mock_df_msft]).sort_index()
        
        self.mock_market_data.fetch_historical_stock_data.return_value = mock_combined_df

        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()
        
        df = self.data_handler.fetch_historical_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            flatten=False # Explicitly pass flatten=False
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(df.index.nlevels, 2)
        self.assertCountEqual(df.index.get_level_values('symbol').unique(), symbols)
        self.mock_market_data.fetch_historical_stock_data.assert_called_once()
        # DataHandler explicitly passed flatten=False, so assertion should include it.
        # Still no 'timeframe' because DataHandler doesn't explicitly pass it.
        self.mock_market_data.fetch_historical_stock_data.assert_called_once_with(
            symbols=symbols, start_date=start_date, end_date=end_date, flatten=False
        )


    def test_fetch_historical_stock_data_empty_response(self):
        """Test handling of an empty historical data response."""
        # return_value is already an empty DataFrame by default in setUp
        
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()
        
        df = self.data_handler.fetch_historical_stock_data(
            symbols="NONEXISTENT",
            start_date=start_date,
            end_date=end_date
            # No 'flatten' argument explicitly passed here. It will rely on MarketDataClient's default.
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)
        self.mock_market_data.fetch_historical_stock_data.assert_called_once()
        # FIX: No 'flatten' argument in the assertion because DataHandler didn't pass it.
        # It relies on MarketDataClient's default, which the mock doesn't see as an explicit argument.
        self.mock_market_data.fetch_historical_stock_data.assert_called_once_with(
            symbols="NONEXISTENT", start_date=start_date, end_date=end_date
        )


    def test_fetch_historical_stock_data_api_error(self):
        """
        Test handling of an API exception during historical data fetch.
        If MarketDataClient's mock raises an exception, DataHandler's delegation
        will cause the exception to propagate, so we assert for that.
        """
        self.mock_market_data.fetch_historical_stock_data.side_effect = Exception("Alpaca connection error")

        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()
        
        with self.assertRaises(Exception) as cm:
            self.data_handler.fetch_historical_stock_data(
                symbols="AAPL",
                start_date=start_date,
                end_date=end_date
                # No 'flatten' argument explicitly passed here. It will rely on MarketDataClient's default.
            )
        self.assertIn("Alpaca connection error", str(cm.exception))
        
        self.mock_market_data.fetch_historical_stock_data.assert_called_once()
        # FIX: No 'flatten' argument in the assertion because DataHandler didn't pass it.
        self.mock_market_data.fetch_historical_stock_data.assert_called_once_with(
            symbols="AAPL", start_date=start_date, end_date=end_date
        )

    # --- Tests for account and positions ---

    def test_get_account_details_success(self):
        """Test successful fetching of account details."""
        account_details = self.data_handler.get_account_details()
        self.assertIsNotNone(account_details)
        self.assertEqual(account_details.status, 'ACTIVE')
        self.assertEqual(account_details.buying_power, '200000')
        self.mock_brokerage.get_account_details.assert_called_once()

    def test_get_account_details_api_error(self):
        """
        Test handling of an API exception during account details fetch.
        If BrokerageClient's mock raises an exception, DataHandler's delegation
        will cause the exception to propagate, so we assert for that.
        """
        self.mock_brokerage.get_account_details.side_effect = Exception("Account access denied")
        
        with self.assertRaises(Exception) as cm:
            self.data_handler.get_account_details()
        self.assertIn("Account access denied", str(cm.exception))

        self.mock_brokerage.get_account_details.assert_called_once()


    def test_get_all_positions_success_empty(self):
        """Test fetching all positions when none exist."""
        positions = self.data_handler.get_all_positions()
        self.assertIsInstance(positions, list)
        self.assertTrue(len(positions) == 0)
        self.mock_brokerage.get_all_positions.assert_called_once()

    def test_get_all_positions_success_with_positions(self):
        """Test fetching all positions when some exist."""
        mock_position1 = MagicMock(symbol="AAPL", qty="10", current_price="150.0")
        mock_position2 = MagicMock(symbol="MSFT", qty="5", current_price="200.0")
        self.mock_brokerage.get_all_positions.return_value = [mock_position1, mock_position2]

        positions = self.data_handler.get_all_positions()
        self.assertIsInstance(positions, list)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].symbol, "AAPL")
        self.mock_brokerage.get_all_positions.assert_called_once()
    
    # --- Tests for trading (submit_order, etc.) ---

    def test_submit_market_order_success(self):
        """Test that submit_market_order calls the underlying client correctly."""
        order_result = self.data_handler.submit_market_order(
            symbol="TEST", qty=1.0, side=OrderSide.BUY
        )
        self.assertIsNotNone(order_result)
        self.mock_brokerage.submit_market_order.assert_called_once_with(
            "TEST", 1.0, OrderSide.BUY
        )
        self.assertEqual(order_result.client_order_id, 'mock_order_id_123')
        self.assertEqual(order_result.status, 'accepted')


    def test_submit_market_order_api_error(self):
        """
        Test handling of an API exception during order submission.
        If BrokerageClient's mock raises an exception, DataHandler's delegation
        will cause the exception to propagate, so we assert for that.
        """
        self.mock_brokerage.submit_market_order.side_effect = Exception("Insufficient funds")
        
        with self.assertRaises(Exception) as cm:
            self.data_handler.submit_market_order(
                symbol="TEST", qty=1.0, side=OrderSide.BUY
            )
        self.assertIn("Insufficient funds", str(cm.exception))

        self.mock_brokerage.submit_market_order.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
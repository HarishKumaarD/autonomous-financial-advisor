# test_data_handler.py (Corrected)

import unittest
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Import the classes from your module
from afa_core.data_handler import MarketDataClient, BrokerageClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide

# Load environment variables
load_dotenv()

class TestAlpacaClients(unittest.TestCase):
    """
    Test suite for MarketDataClient and BrokerageClient.
    These are integration tests that require live paper trading API keys.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the clients once for all tests.
        Skips all tests if API keys are not found.
        """
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")

        if not api_key or not secret_key:
            raise unittest.SkipTest("Alpaca API keys not found in .env file. Skipping tests.")

        cls.market_client = MarketDataClient(api_key, secret_key)
        cls.brokerage_client = BrokerageClient(api_key, secret_key, paper=True)
        print("Clients initialized for testing.")

    def test_01_fetch_single_symbol_flattened(self):
        """
        Tests fetching data for a single symbol with flatten=True.
        """
        print("\nRunning test_01_fetch_single_symbol_flattened...")
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        df = self.market_client.fetch_historical_stock_data(
            symbols="AAPL",
            start_date=start_date,
            end_date=end_date,
            timeframe=TimeFrame.Day,
            flatten=True
        )

        self.assertIsInstance(df, pd.DataFrame, "Should return a pandas DataFrame.")
        self.assertFalse(df.empty, "DataFrame should not be empty.")
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index should be a DatetimeIndex.")
        self.assertNotIn('symbol', df.index.names, "Symbol should not be in the index when flattened.")
        print("...PASSED")

    def test_02_fetch_multiple_symbols(self):
        """
        Tests fetching data for multiple symbols, resulting in a MultiIndex DataFrame.
        """
        print("\nRunning test_02_fetch_multiple_symbols...")
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        df = self.market_client.fetch_historical_stock_data(
            symbols=["GOOGL", "MSFT"],
            start_date=start_date,
            end_date=end_date,
            timeframe=TimeFrame.Day,
            flatten=False # Default behavior for multiple symbols
        )

        self.assertIsInstance(df, pd.DataFrame, "Should return a pandas DataFrame.")
        self.assertFalse(df.empty, "DataFrame should not be empty.")
        self.assertIsInstance(df.index, pd.MultiIndex, "Index should be a MultiIndex.")
        self.assertIn('symbol', df.index.names, "Symbol should be in the index for multiple symbols.")
        print("...PASSED")

    def test_03_get_account_details(self):
        """
        Tests retrieving account details from the brokerage.
        """
        print("\nRunning test_03_get_account_details...")
        account = self.brokerage_client.get_account_details()
        
        self.assertIsNotNone(account, "Should return an account object, not None.")
        self.assertTrue(hasattr(account, 'equity'), "Account object should have an 'equity' attribute.")
        print("...PASSED")

    def test_04_submit_and_cancel_order(self):
        """
        Tests submitting a market order and then cancelling all open orders.
        """
        print("\nRunning test_04_submit_and_cancel_order...")
        
        # 1. Submit a test buy order
        order = self.brokerage_client.submit_market_order(
            symbol="F", 
            qty=1.0,
            side=OrderSide.BUY
        )

        self.assertIsNotNone(order, "Order object should not be None after submission.")
        self.assertEqual(order.symbol, "F", "Order symbol should be F.")
        
        # --- FIX IS HERE ---
        # Convert the order.qty string to a float before comparing
        self.assertEqual(float(order.qty), 1.0, "Order quantity should be 1.0.")

        # 2. Now cancel all open orders
        cancel_statuses = self.brokerage_client.cancel_all_orders()
        self.assertIsInstance(cancel_statuses, list, "cancel_all_orders should return a list.")
        print("...PASSED")

if __name__ == "__main__":
    unittest.main(verbosity=2)
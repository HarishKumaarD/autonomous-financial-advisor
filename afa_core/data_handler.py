# afa_core/data_handler.py 

import os
import logging
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MarketDataClient:
    """Handles fetching of historical market data from Alpaca."""

    def __init__(self, api_key: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def fetch_historical_stock_data(
        self,
        symbols: str | list[str],
        start_date,
        end_date,
        timeframe: TimeFrame = TimeFrame.Day,
        flatten: bool = True
    ) -> pd.DataFrame:
        """
        Fetches historical stock bar data from Alpaca.

        Args:
            symbols (str or list): Stock symbol(s) to fetch data for
            start_date (datetime): Start date for historical data
            end_date (datetime): End date for historical data
            timeframe (TimeFrame): Data timeframe (default: daily)
            flatten (bool): If True and only one symbol, returns flat DataFrame

        Returns:
            pandas.DataFrame: Historical stock data with lowercase OHLCV column names.
        """
        try:
            # Determine if we're dealing with multiple symbols
            is_multi_symbol = isinstance(symbols, list) and len(symbols) > 1
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            bars = self.client.get_stock_bars(request_params)
            df = bars.df

            if df.empty:
                logging.warning(f"No historical data returned from Alpaca for {symbols}.")
                return df

            # Ensure consistent lowercase column names immediately after fetching
            df.columns = df.columns.str.lower()

            # Handle MultiIndex logic
            if isinstance(df.index, pd.MultiIndex):
                # Ensure proper index names for SignalGenerator compatibility
                if df.index.names != ['symbol', 'timestamp']:
                    # Try to fix the index names
                    if len(df.index.names) == 2:
                        df.index.names = ['symbol', 'timestamp']
                    
                # Only flatten for single symbol when requested
                if flatten and isinstance(symbols, str):
                    # Single symbol - flatten the MultiIndex
                    df = df.reset_index(level="symbol", drop=True)
                elif is_multi_symbol:
                    # Multiple symbols - keep MultiIndex but ensure proper ordering
                    if df.index.names == ['symbol', 'timestamp']:
                        df = df.sort_index()
                    else:
                        # Try to reorder if needed
                        try:
                            df = df.reorder_levels(['symbol', 'timestamp']).sort_index()
                        except:
                            logging.warning("Could not reorder MultiIndex levels")
            else:
                # Single index - ensure it's named 'timestamp' for consistency
                if df.index.name != 'timestamp':
                    df.index.name = 'timestamp'
            
            return df

        except APIError as e:
            logging.error(f"Alpaca API Error while fetching data for {symbols}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {symbols}: {e}")
            return pd.DataFrame()


class BrokerageClient:
    """Handles trading-related operations with Alpaca."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account_details(self):
        try:
            return self.client.get_account()
        except Exception as e:
            logging.error(f"Failed to fetch account details: {e}")
            return None

    def submit_market_order(self, symbol: str, qty: float, side: OrderSide):
        try:
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            return self.client.submit_order(order_data=market_order_data)
        except Exception as e:
            logging.error(f"Failed to submit market order for {symbol}: {e}")
            return None

    def get_all_positions(self):
        try:
            return self.client.get_all_positions()
        except Exception as e:
            logging.error(f"Failed to fetch positions: {e}")
            return []

    def get_orders(self, status: str = None):
        try:
            return self.client.get_orders(filter=status)
        except Exception as e:
            logging.error(f"Failed to fetch orders: {e}")
            return []

    def cancel_all_orders(self):
        try:
            return self.client.cancel_orders()
        except Exception as e:
            logging.error(f"Failed to cancel orders: {e}")
            return []

    def get_portfolio_history(self):
        try:
            return self.client.get_portfolio_history()
        except Exception as e:
            logging.error(f"Failed to fetch portfolio history: {e}")
            return None


class DataHandler:
    """
    Wrapper class to provide unified access to MarketDataClient and BrokerageClient.
    Keeps the original API for backward compatibility.
    """

    def __init__(self):
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("API keys must be set as environment variables.")

        self.market_data = MarketDataClient(self.api_key, self.secret_key)
        self.brokerage = BrokerageClient(self.api_key, self.secret_key, paper=True)

    # Market Data
    def fetch_historical_stock_data(self, *args, **kwargs):
        # Delegate to the MarketDataClient
        return self.market_data.fetch_historical_stock_data(*args, **kwargs)

    # Brokerage
    def get_account_details(self):
        return self.brokerage.get_account_details()

    def submit_market_order(self, symbol, qty, side):
        return self.brokerage.submit_market_order(symbol, qty, side)

    def get_all_positions(self):
        return self.brokerage.get_all_positions()

    def get_orders(self, status=None):
        return self.brokerage.get_orders(status)

    def cancel_all_orders(self):
        return self.brokerage.cancel_all_orders()

    def get_portfolio_history(self):
        return self.brokerage.get_portfolio_history()
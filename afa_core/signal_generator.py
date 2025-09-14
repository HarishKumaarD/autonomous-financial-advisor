# afa_core/signal_generator.py 

import pandas as pd
import numpy as np
import ta  # using "ta" library for technical analysis


class SignalGenerator:
    """
    Generates trading signals from classical technical indicators.
    Supports both single-symbol OHLCV DataFrames and multi-symbol (MultiIndex).
    """

    def __init__(self, df_ohlcv: pd.DataFrame):
        """
        Initializes the SignalGenerator with historical data.

        Args:
            df_ohlcv (pd.DataFrame): Input DataFrame.
                                    For single symbol: DatetimeIndex (name 'timestamp'),
                                                        columns 'open', 'high', 'low', 'close'.
                                    For multi-symbol: MultiIndex (levels 'symbol', 'timestamp'),
                                                    columns 'open', 'high', 'low', 'close'.
        Raises:
            TypeError: If input is not a DataFrame.
            ValueError: If DataFrame is empty or missing required OHLCV columns.
                        If MultiIndex is detected but doesn't have 2 levels.
        """
        # print("\n[SignalGenerator.__init__] - START") # Keep or remove these debug prints as needed
        if not isinstance(df_ohlcv, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if df_ohlcv.empty:
            self.df = pd.DataFrame()
            self.multi_symbol = False
            self.added_signals = []
            # print("[SignalGenerator.__init__] - Initialized with empty DataFrame.")
            return

        # Crucial: Ensure a deep copy to prevent external modifications
        self.df = df_ohlcv.copy(deep=True)
        self.df.columns = self.df.columns.str.lower() # Normalize column names
        # print(f"[SignalGenerator.__init__] - Input df index before processing: {self.df.index.names}, type: {type(self.df.index)}")


        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Input DataFrame must contain columns: {', '.join(required_cols)}.")

        # Determine multi-symbol mode and validate index structure
        if isinstance(self.df.index, pd.MultiIndex):
            self.multi_symbol = True
            # print(f"[SignalGenerator.__init__] - Detected MultiIndex. N levels: {self.df.index.nlevels}")
            if self.df.index.nlevels != 2:
                raise ValueError(f"MultiIndex DataFrame expected 2 levels, but found {self.df.index.nlevels}.")
            
            # Defensive check and reordering if levels are not ['symbol', 'timestamp']
            if self.df.index.names != ['symbol', 'timestamp']:
                current_names = list(self.df.index.names)
                # print(f"[SignalGenerator.__init__] - MultiIndex has unexpected names: {current_names}. Attempting normalization.")
                
                if 'symbol' in current_names and 'timestamp' in current_names:
                    if current_names.index('symbol') != 0 or current_names.index('timestamp') != 1:
                        try:
                            self.df = self.df.reorder_levels(['symbol', 'timestamp']).sort_index()
                            # print(f"[SignalGenerator.__init__] - Info: MultiIndex levels reordered to {self.df.index.names}")
                        except Exception as e:
                            print(f"[SignalGenerator.__init__] - Warning: Failed to reorder MultiIndex levels to ['symbol', 'timestamp']. Error: {e}. Current names: {self.df.index.names}")
                else:
                    if current_names[0] is None and current_names[1] is None:
                        self.df.index.set_names(['symbol', 'timestamp'], inplace=True)
                        # print(f"[SignalGenerator.__init__] - Info: MultiIndex levels set names to {self.df.index.names}")
                    elif current_names[0] == 'symbol' and current_names[1] is None:
                        self.df.index.set_names(['symbol', 'timestamp'], level=1, inplace=True)
                        # print(f"[SignalGenerator.__init__] - Info: MultiIndex level 1 name set to 'timestamp'. Current names: {self.df.index.names}")
                    elif current_names[0] is None and current_names[1] == 'timestamp':
                        self.df.index.set_names(['symbol', 'timestamp'], level=0, inplace=True)
                        # print(f"[SignalGenerator.__init__] - Info: MultiIndex level 0 name set to 'symbol'. Current names: {self.df.index.names}")
                    else:
                        # print(f"[SignalGenerator.__init__] - Warning: MultiIndex levels have unexpected names {self.df.index.names}. Attempting to assign 'symbol' and 'timestamp'.")
                        try:
                            self.df.index.set_names(['symbol', 'timestamp'], inplace=True)
                            self.df = self.df.reorder_levels(['symbol', 'timestamp']).sort_index()
                            # print(f"[SignalGenerator.__init__] - Info: MultiIndex levels assigned and reordered to {self.df.index.names}")
                        except Exception as e:
                            print(f"[SignalGenerator.__init__] - Warning: Final attempt to normalize MultiIndex failed. Error: {e}")
            
            # Final check of index names after __init__ logic
            if self.df.index.names != ['symbol', 'timestamp']:
                print(f"[SignalGenerator.__init__] - CRITICAL: MultiIndex names are still not ['symbol', 'timestamp'] after init: {self.df.index.names}")
                # Consider raising an error here if this state is unacceptable
                
        else: # Single symbol DataFrame
            self.multi_symbol = False
            # print(f"[SignalGenerator.__init__] - Detected single Index. Type: {type(self.df.index)}")
            if not isinstance(self.df.index, pd.DatetimeIndex):
                raise TypeError("Single-symbol DataFrame must have a DatetimeIndex.")
            if self.df.index.name != 'timestamp':
                self.df.index.name = 'timestamp'
                # print(f"[SignalGenerator.__init__] - Info: Single Index name set to {self.df.index.name}")
        
        self.added_signals = [] # List to keep track of added signal columns
        # print(f"[SignalGenerator.__init__] - END. Final self.df index: {self.df.index.names}, multi_symbol: {self.multi_symbol}")

    # ---------- HELPER ----------
    def _apply_per_symbol(self, func, *args, **kwargs):
        """
        Apply a function either per symbol (if MultiIndex) or directly.
        When in multi-symbol mode, it groups by 'symbol' and applies the
        function to each symbol's data, which is temporarily single-indexed.
        """
        # print(f"\n[SignalGenerator._apply_per_symbol] - START. Current self.df index: {self.df.index.names}, multi_symbol: {self.multi_symbol}")
        
        if self.multi_symbol:
            if 'symbol' not in self.df.index.names:
                # print(f"[SignalGenerator._apply_per_symbol] - ERROR: 'symbol' not in MultiIndex levels: {self.df.index.names}")
                raise ValueError(f"Cannot group by 'symbol'. MultiIndex levels are: {self.df.index.names}")
            
            # The key change is how we process the groups and then concatenate
            processed_dfs = []
            for symbol, group_df in self.df.groupby(level="symbol"):
                # Pass the single-indexed DataFrame to the function
                temp_df = func(group_df.droplevel("symbol"), *args, **kwargs)
                
                # Re-add the symbol level. Create a new MultiIndex.
                # Ensure the temporary df has a 'timestamp' index before re-indexing
                if not isinstance(temp_df.index, pd.DatetimeIndex) or temp_df.index.name != 'timestamp':
                    if temp_df.index.name is None: # If name is lost, try to restore
                        temp_df.index.name = 'timestamp'
                    else:
                        print(f"[SignalGenerator._apply_per_symbol] - WARNING: Applied function returned DataFrame with unexpected index type or name: {type(temp_df.index)}, {temp_df.index.name}. Attempting to force 'timestamp'.")
                        # You might need more robust handling here if func radically changes index
                        # For now, we assume it's a DatetimeIndex that might have lost its name.
                        temp_df = temp_df.set_index(pd.to_datetime(temp_df.index), drop=False)
                        temp_df.index.name = 'timestamp'


                # Create a MultiIndex for this processed group
                multi_idx = pd.MultiIndex.from_product([[symbol], temp_df.index], names=['symbol', 'timestamp'])
                temp_df.index = multi_idx
                processed_dfs.append(temp_df)
            
            # Concatenate all processed (and re-indexed) DataFrames
            result_df = pd.concat(processed_dfs).sort_index()
            
            # print(f"[SignalGenerator._apply_per_symbol] - MultiIndex reconstruction END. Result df index: {result_df.index.names}, type: {type(result_df.index)}")
            # print(f"[SignalGenerator._apply_per_symbol] - END. Returning DataFrame with index: {result_df.index.names}")
            return result_df
        else:
            # print("[SignalGenerator._apply_per_symbol] - END. Applying function directly to single-symbol DataFrame.")
            return func(self.df, *args, **kwargs)

    # ---------- INDICATORS ----------
    # (No functional changes to indicator methods - they call _apply_per_symbol)

    def add_sma_crossover_signal(self, short_window=50, long_window=200):
        # print(f"[SignalGenerator.add_sma_crossover_signal] - START. self.df index: {self.df.index.names}")
        # Re-introducing the inner _sma function for clarity as the assign chain makes debugging harder
        def _sma(df, short_window, long_window):
            if len(df) < long_window: # Not enough data for long SMA
                df_copy = df.copy() # Operate on a copy to avoid SettingWithCopyWarning
                df_copy[f"sma_{short_window}"] = np.nan
                df_copy[f"sma_{long_window}"] = np.nan
                df_copy["sma_signal"] = 0.0
                df_copy["sma_crossover"] = 0.0
                return df_copy

            df_copy = df.copy()
            df_copy[f"sma_{short_window}"] = df_copy["close"].rolling(short_window).mean()
            df_copy[f"sma_{long_window}"] = df_copy["close"].rolling(long_window).mean()
            df_copy["sma_signal"] = np.where(
                df_copy[f"sma_{short_window}"] > df_copy[f"sma_{long_window}"], 1.0, 0.0
            )
            df_copy["sma_crossover"] = df_copy["sma_signal"].diff().fillna(0)
            return df_copy

        self.df = self._apply_per_symbol(_sma, short_window, long_window)
        self.added_signals.extend([f"sma_{short_window}", f"sma_{long_window}", "sma_signal", "sma_crossover"])
        self.added_signals = list(set(self.added_signals)) 
        # print(f"[SignalGenerator.add_sma_crossover_signal] - END. self.df index: {self.df.index.names}")
        return self.df

    def add_rsi_signal(self, length=14, upper_bound=70, lower_bound=30):
        # print(f"[SignalGenerator.add_rsi_signal] - START. self.df index: {self.df.index.names}")
        def _rsi(df, length, upper, lower):
            if len(df) < length: 
                df_copy = df.copy()
                df_copy["rsi"] = np.nan
                df_copy["rsi_signal"] = 0
                return df_copy

            df_copy = df.copy()
            rsi = ta.momentum.RSIIndicator(close=df_copy["close"], window=length).rsi()
            df_copy["rsi"] = rsi
            df_copy["rsi_signal"] = 0
            df_copy.loc[rsi > upper, "rsi_signal"] = -1 
            df_copy.loc[rsi < lower, "rsi_signal"] = 1  
            return df_copy

        self.df = self._apply_per_symbol(_rsi, length, upper_bound, lower_bound)
        self.added_signals.extend(["rsi", "rsi_signal"])
        self.added_signals = list(set(self.added_signals))
        # print(f"[SignalGenerator.add_rsi_signal] - END. self.df index: {self.df.index.names}")
        return self.df

    def add_macd_signal(self, fast=12, slow=26, signal=9):
        # print(f"[SignalGenerator.add_macd_signal] - START. self.df index: {self.df.index.names}")
        def _macd(df, fast, slow, signal):
            if len(df) < slow + signal: 
                df_copy = df.copy()
                df_copy["macd"] = np.nan
                df_copy["macd_signal_line"] = np.nan
                df_copy["macd_hist"] = np.nan
                df_copy["macd_signal"] = 0.0
                df_copy["macd_crossover"] = 0.0
                return df_copy

            df_copy = df.copy()
            macd_ind = ta.trend.MACD(df_copy["close"], window_slow=slow, window_fast=fast, window_sign=signal)
            df_copy["macd"] = macd_ind.macd()
            df_copy["macd_signal_line"] = macd_ind.macd_signal()
            df_copy["macd_hist"] = macd_ind.macd_diff() 

            df_copy["macd_signal"] = np.where(df_copy["macd"] > df_copy["macd_signal_line"], 1.0, -1.0)
            df_copy["macd_crossover"] = df_copy["macd_signal"].diff().fillna(0)
            return df_copy

        self.df = self._apply_per_symbol(_macd, fast, slow, signal)
        self.added_signals.extend(["macd", "macd_signal_line", "macd_hist", "macd_signal", "macd_crossover"])
        self.added_signals = list(set(self.added_signals))
        # print(f"[SignalGenerator.add_macd_signal] - END. self.df index: {self.df.index.names}")
        return self.df

    def add_bollinger_bands_signal(self, length=20, std_dev=2):
        # print(f"[SignalGenerator.add_bollinger_bands_signal] - START. self.df index: {self.df.index.names}")
        def _bb(df, length, std_dev):
            if len(df) < length: 
                df_copy = df.copy()
                df_copy["bb_bbm"] = np.nan
                df_copy["bb_bbh"] = np.nan
                df_copy["bb_bbl"] = np.nan
                df_copy["bb_signal"] = 0
                return df_copy

            df_copy = df.copy()
            bb = ta.volatility.BollingerBands(close=df_copy["close"], window=length, window_dev=std_dev)
            df_copy["bb_bbm"] = bb.bollinger_mavg()
            df_copy["bb_bbh"] = bb.bollinger_hband() 
            df_copy["bb_bbl"] = bb.bollinger_lband() 

            df_copy["bb_signal"] = 0
            df_copy.loc[df_copy["close"] <= df_copy["bb_bbl"], "bb_signal"] = 1  
            df_copy.loc[df_copy["close"] >= df_copy["bb_bbh"], "bb_signal"] = -1 
            return df_copy

        self.df = self._apply_per_symbol(_bb, length, std_dev)
        self.added_signals.extend(["bb_bbm", "bb_bbh", "bb_bbl", "bb_signal"])
        self.added_signals = list(set(self.added_signals))
        # print(f"[SignalGenerator.add_bollinger_bands_signal] - END. self.df index: {self.df.index.names}")
        return self.df

    def add_stochastic_signal(self, k_window=14, d_window=3):
        # print(f"[SignalGenerator.add_stochastic_signal] - START. self.df index: {self.df.index.names}")
        def _stoch(df, k, d):
            if len(df) < k: 
                df_copy = df.copy()
                df_copy["stoch_k"] = np.nan
                df_copy["stoch_d"] = np.nan
                df_copy["stoch_signal"] = 0
                return df_copy

            df_copy = df.copy()
            stoch = ta.momentum.StochasticOscillator(
                high=df_copy["high"], low=df_copy["low"], close=df_copy["close"], window=k, smooth_window=d
            )
            df_copy["stoch_k"] = stoch.stoch()
            df_copy["stoch_d"] = stoch.stoch_signal() 

            df_copy["stoch_signal"] = 0
            buy_cond = (df_copy["stoch_k"].shift(1) < df_copy["stoch_d"].shift(1)) & \
                        (df_copy["stoch_k"] > df_copy["stoch_d"]) & \
                        (df_copy["stoch_k"] < 20)
            sell_cond = (df_copy["stoch_k"].shift(1) > df_copy["stoch_d"].shift(1)) & \
                        (df_copy["stoch_k"] < df_copy["stoch_d"]) & \
                        (df_copy["stoch_k"] > 80)
            
            df_copy.loc[buy_cond, "stoch_signal"] = 1
            df_copy.loc[sell_cond, "stoch_signal"] = -1
            return df_copy

        self.df = self._apply_per_symbol(_stoch, k_window, d_window)
        self.added_signals.extend(["stoch_k", "stoch_d", "stoch_signal"])
        self.added_signals = list(set(self.added_signals))
        # print(f"[SignalGenerator.add_stochastic_signal] - END. self.df index: {self.df.index.names}")
        return self.df


    # ---------- COMPOSITE SIGNAL ----------
    def add_composite_signal(self, weights=None, buy_threshold=0.3, sell_threshold=-0.3):
        # print(f"[SignalGenerator.add_composite_signal] - START. self.df index: {self.df.index.names}")
        if weights is None:
            weights = {
                "sma_signal": 0.3,
                "rsi_signal": 0.2,
                "macd_signal": 0.2,
                "bb_signal": 0.15,
                "stoch_signal": 0.15,
            }

        def _composite(df, weights, buy_thr, sell_thr):
            df_copy = df.copy()
            df_copy["composite_signal_raw"] = 0.0 
            for sig, w in weights.items():
                if sig in df_copy.columns:
                    df_copy["composite_signal_raw"] += df_copy[sig].fillna(0) * w
            
            df_copy["composite_action"] = 0
            df_copy.loc[df_copy["composite_signal_raw"] > buy_thr, "composite_action"] = 1
            df_copy.loc[df_copy["composite_signal_raw"] < sell_thr, "composite_action"] = -1
            return df_copy

        self.df = self._apply_per_symbol(_composite, weights, buy_threshold, sell_threshold)
        self.added_signals.extend(["composite_signal_raw", "composite_action"])
        self.added_signals = list(set(self.added_signals))
        # print(f"[SignalGenerator.add_composite_signal] - END. self.df index: {self.df.index.names}")
        return self.df

    # ---------- UTILS ----------
    def get_current_signals(self, symbol=None):
        if self.df.empty:
            return {}

        if self.multi_symbol:
            if symbol is None:
                all_latest_signals = {}
                for s in self.df.index.get_level_values('symbol').unique():
                    try:
                        latest_row = self.df.xs(s, level="symbol").iloc[-1]
                        all_latest_signals[s] = {
                            col: latest_row[col]
                            for col in latest_row.index
                            if col in self.added_signals or "action" in col 
                        }
                    except IndexError: 
                        all_latest_signals[s] = {}
                return all_latest_signals
            else:
                if symbol not in self.df.index.get_level_values('symbol').unique():
                    return {}
                try:
                    latest_row = self.df.xs(symbol, level="symbol").iloc[-1]
                    return {
                        col: latest_row[col]
                        for col in latest_row.index
                        if col in self.added_signals or "action" in col 
                    }
                except IndexError:
                    return {}
        else: # Single symbol mode
            latest_row = self.df.iloc[-1]
            return {
                col: latest_row[col]
                for col in latest_row.index
                if col in self.added_signals or "action" in col 
            }

    def get_signals_summary(self, symbol=None):
        if self.df.empty:
            return {}

        summary_data = {}
        target_df = self.df

        if self.multi_symbol:
            if symbol is None:
                for s in target_df.index.get_level_values('symbol').unique():
                    df_symbol = target_df.xs(s, level="symbol")
                    symbol_summary = {}
                    for col in df_symbol.columns:
                        if (("signal" in col or "action" in col) and col in self.added_signals) or col == "composite_action":
                            cleaned_col = df_symbol[col].fillna(0) 
                            symbol_summary[col] = {
                                "buy_signals": (cleaned_col == 1).sum(),
                                "sell_signals": (cleaned_col == -1).sum(),
                                "hold_signals": (cleaned_col == 0).sum(),
                                "current_signal": cleaned_col.iloc[-1] if len(cleaned_col) > 0 else 0,
                            }
                    summary_data[s] = symbol_summary
                return summary_data
            else:
                if symbol not in target_df.index.get_level_values('symbol').unique():
                    return {}
                target_df = target_df.xs(symbol, level="symbol")

        for col in target_df.columns:
            if (("signal" in col or "action" in col) and col in self.added_signals) or col == "composite_action":
                cleaned_col = target_df[col].fillna(0) 
                summary_data[col] = {
                    "buy_signals": (cleaned_col == 1).sum(),
                    "sell_signals": (cleaned_col == -1).sum(),
                    "hold_signals": (cleaned_col == 0).sum(),
                    "current_signal": cleaned_col.iloc[-1] if len(cleaned_col) > 0 else 0,
                }
        return summary_data
# test_signals.py
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from afa_core.data_handler import DataHandler
from afa_core.signal_generator import SignalGenerator

def test_signal_generator():
    """Test the SignalGenerator functionality"""
    try:
        # Get historical data
        print("🔄 Fetching historical data...")
        handler = DataHandler()
        
        # Get 6 months of data to have enough for long-term indicators (200 SMA)
        end_date = datetime.now() - timedelta(days=1)  # Yesterday to avoid incomplete data
        start_date = end_date - timedelta(days=180)    # 6 months for 200-day SMA
        
        data = handler.fetch_historical_stock_data(
            symbols="AAPL",
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"✅ Fetched {len(data)} bars of data")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        print(f"📊 Columns: {list(data.columns)}")
        
        # Check if we have enough data
        if len(data) < 50:
            print("⚠️  Warning: Limited data available, some indicators may not work properly")
        
        # Initialize Signal Generator
        print("\n🔄 Initializing Signal Generator...")
        signal_gen = SignalGenerator(data)
        print("✅ Signal Generator initialized!")
        
        # Add various technical indicators
        print("\n🔄 Generating technical signals...")
        
        # SMA Crossover - using shorter windows if we don't have enough data
        data_length = len(data)
        if data_length >= 200:
            short_window, long_window = 50, 200
        elif data_length >= 50:
            short_window, long_window = 20, 50
        else:
            short_window, long_window = 5, 20
            
        print(f"   Using SMA windows: {short_window}/{long_window}")
        signal_gen.add_sma_crossover_signal(short_window=short_window, long_window=long_window)
        print("✅ SMA crossover signals added")
        
        # RSI
        signal_gen.add_rsi_signal()
        print("✅ RSI signals added")
        
        # MACD
        signal_gen.add_macd_signal()
        print("✅ MACD signals added")
        
        # Bollinger Bands
        signal_gen.add_bollinger_bands_signal()
        print("✅ Bollinger Bands signals added")
        
        # Stochastic
        signal_gen.add_stochastic_signal()
        print("✅ Stochastic signals added")
        
        # Composite signal
        signal_gen.add_composite_signal()
        print("✅ Composite signal added")
        
        # Get current signals
        print("\n📊 Current Signals:")
        current_signals = signal_gen.get_current_signals()
        
        if current_signals:
            for signal_name, value in current_signals.items():
                if pd.isna(value):
                    emoji = "⚪"
                    value_str = "N/A"
                elif value > 0:
                    emoji = "🟢"
                    value_str = f"{value:.2f}"
                elif value < 0:
                    emoji = "🔴"
                    value_str = f"{value:.2f}"
                else:
                    emoji = "🟡"
                    value_str = f"{value:.2f}"
                print(f"   {emoji} {signal_name}: {value_str}")
        else:
            print("   ⚠️  No signals available")
        
        # Get signals summary
        print("\n📈 Signals Summary:")
        summary = signal_gen.get_signals_summary()
        
        if summary:
            for signal_name, stats in summary.items():
                current_val = stats['current_signal']
                current_str = "N/A" if pd.isna(current_val) else f"{current_val:.2f}"
                print(f"   📊 {signal_name}:")
                print(f"      Buy: {stats['buy_signals']}, Sell: {stats['sell_signals']}, Hold: {stats['hold_signals']}")
                print(f"      Current: {current_str}")
        else:
            print("   ⚠️  No summary data available")
        
        # Show recent data with signals
        print("\n📋 Recent Data with Signals (last 5 rows):")
        
        # Define preferred columns to show
        base_columns = ['close']
        indicator_columns = ['rsi', f'sma_{short_window}', f'sma_{long_window}', 'macd', 'bb_bbm']
        signal_columns = ['sma_signal', 'rsi_signal', 'macd_signal', 'bb_signal', 'composite_action']
        
        # Get available columns from each category
        available_base = [col for col in base_columns if col in signal_gen.df.columns]
        available_indicators = [col for col in indicator_columns if col in signal_gen.df.columns]
        available_signals = [col for col in signal_columns if col in signal_gen.df.columns]
        
        # Combine available columns
        display_columns = available_base + available_indicators + available_signals
        
        if display_columns:
            recent_data = signal_gen.df[display_columns].tail()
            # Round numeric columns for better display
            recent_data = recent_data.round(4)
            print(recent_data.to_string())
        else:
            print("   ⚠️  No display columns available")
        
        # Additional validation
        print("\n🔍 Data Validation:")
        total_rows = len(signal_gen.df)
        print(f"   Total rows: {total_rows}")
        
        # Check for NaN values in key signals
        key_signals = ['composite_action', 'sma_signal', 'rsi_signal']
        for signal in key_signals:
            if signal in signal_gen.df.columns:
                nan_count = signal_gen.df[signal].isna().sum()
                valid_count = total_rows - nan_count
                print(f"   {signal}: {valid_count}/{total_rows} valid values")
        
        # Test trading interpretation
        print("\n🎯 Trading Interpretation:")
        if current_signals and 'composite_action' in current_signals:
            composite_signal = current_signals['composite_action']
            if pd.isna(composite_signal):
                interpretation = "WAIT - Insufficient data for decision"
            elif composite_signal > 0:
                interpretation = "BUY - Bullish signals dominate"
            elif composite_signal < 0:
                interpretation = "SELL - Bearish signals dominate"
            else:
                interpretation = "HOLD - Neutral signals"
            print(f"   🎯 {interpretation}")
        else:
            print("   ⚠️  No composite signal available")
        
        print("\n🎉 Signal generation test completed successfully!")
        
        # Return the signal generator for further testing if needed
        return signal_gen
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install pandas numpy ta yfinance")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multi_symbol():
    """Test SignalGenerator with multiple symbols"""
    print("\n" + "="*50)
    print("🔄 Testing Multi-Symbol Functionality")
    print("="*50)
    
    try:
        handler = DataHandler()
        
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=90)  # 3 months
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        print(f"🔄 Fetching data for symbols: {symbols}")
        
        # --- CRITICAL CHANGE HERE: Fetch all symbols at once ---
        # The data_handler.fetch_historical_stock_data already returns a MultiIndex
        # DataFrame with 'symbol' and 'timestamp' levels when given a list of symbols.
        multi_df = handler.fetch_historical_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            flatten=False # Ensure it returns MultiIndex
        )
        
        if multi_df.empty:
            print("❌ No data available for multi-symbol test")
            return None
        
        # Verify the MultiIndex structure
        if not isinstance(multi_df.index, pd.MultiIndex):
            raise TypeError("Expected MultiIndex DataFrame after fetching multiple symbols.")
        if multi_df.index.names != ['symbol', 'timestamp']:
            # This is a good place to catch if the level names are unexpected
            print(f"⚠️  MultiIndex levels are not as expected: {multi_df.index.names}. Reordering to ['symbol', 'timestamp'].")
            # If for some reason the order isn't symbol, timestamp, reorder it.
            # This is defensive, as Alpaca's client usually returns it correctly.
            multi_df = multi_df.reorder_levels(['symbol', 'timestamp']).sort_index()

        print(f"✅ Created multi-symbol DataFrame: {len(multi_df)} total rows")
        print(f"✅ MultiIndex levels: {multi_df.index.names}")

        # Test SignalGenerator with multi-symbol data
        signal_gen = SignalGenerator(multi_df)
        
        # Add signals
        signal_gen.add_rsi_signal()
        signal_gen.add_sma_crossover_signal(short_window=20, long_window=50)
        signal_gen.add_composite_signal()
        
        # Get current signals for all symbols
        print("\n📊 Current Signals for All Symbols:")
        all_signals = signal_gen.get_current_signals()
        
        for symbol, signals in all_signals.items():
            print(f"\n   🏢 {symbol}:")
            for signal_name, value in signals.items():
                if pd.isna(value):
                    emoji = "⚪"
                    value_str = "N/A"
                elif value > 0:
                    emoji = "🟢"
                    value_str = f"{value:.2f}"
                elif value < 0:
                    emoji = "🔴" 
                    value_str = f"{value:.2f}"
                else:
                    emoji = "🟡"
                    value_str = f"{value:.2f}"
                print(f"      {emoji} {signal_name}: {value_str}")
        
        print("\n🎉 Multi-symbol test completed successfully!")
        return signal_gen
        
    except Exception as e:
        print(f"❌ Multi-symbol test error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting Signal Generator Tests")
    print("="*50)
    
    # Test single symbol
    print("📈 Testing Single Symbol...")
    single_signal_gen = test_signal_generator()
    
    # Test multiple symbols (optional)
    if single_signal_gen is not None:
        print("\n" + "="*50)
        user_input = input("🤔 Would you like to test multi-symbol functionality? (y/N): ").lower().strip()
        if user_input == 'y' or user_input == 'yes':
            multi_signal_gen = test_multi_symbol()
        else:
            print("⏭️  Skipping multi-symbol test")
    
    print("\n✨ All tests completed!")
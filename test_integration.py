# test_integration.py
"""
Integration test for DataHandler and SignalGenerator classes.
Tests the complete workflow from data fetching to signal generation.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afa_core.data_handler import DataHandler, MarketDataClient, BrokerageClient
from afa_core.signal_generator import SignalGenerator

# Load environment variables
load_dotenv()

def test_environment_setup():
    """Test that environment variables are properly configured"""
    print("Testing environment setup...")
    
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Environment variables not found!")
        print("Please create a .env file with:")
        print("APCA_API_KEY_ID=your_api_key_here")
        print("APCA_API_SECRET_KEY=your_secret_key_here")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...")
    print(f"✅ Secret Key found: {secret_key[:8]}...")
    return True

def test_data_handler_initialization():
    """Test DataHandler initialization and component access"""
    print("\n" + "="*60)
    print("Testing DataHandler Initialization")
    print("="*60)
    
    try:
        handler = DataHandler()
        print("✅ DataHandler initialized successfully")
        
        # Test component access
        print(f"✅ MarketDataClient available: {handler.market_data is not None}")
        print(f"✅ BrokerageClient available: {handler.brokerage is not None}")
        
        return handler
    except Exception as e:
        print(f"❌ DataHandler initialization failed: {e}")
        return None

def test_account_access(handler):
    """Test account access and basic brokerage functionality"""
    print("\n" + "="*60)
    print("Testing Account Access")
    print("="*60)
    
    try:
        account = handler.get_account_details()
        if account:
            print("✅ Account details retrieved successfully")
            print(f"   Account ID: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Cash: ${float(account.cash):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            return True
        else:
            print("⚠️ Could not retrieve account details (may be expected in some environments)")
            return False
    except Exception as e:
        print(f"⚠️ Account access failed: {e}")
        return False

def test_single_symbol_data_and_signals(handler):
    """Test single symbol data fetching and signal generation"""
    print("\n" + "="*60)
    print("Testing Single Symbol: Data + Signals")
    print("="*60)
    
    try:
        # Define date range
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=90)
        
        symbol = "AAPL"
        print(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Fetch data using DataHandler
        data = handler.fetch_historical_stock_data(
            symbols=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            print(f"❌ No data returned for {symbol}")
            return None
        
        print(f"✅ Fetched {len(data)} bars of data")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Index type: {type(data.index)}")
        print(f"   Index name: {data.index.name}")
        
        # Display sample data
        print("\n📊 Sample data (first 3 rows):")
        print(data.head(3).round(2))
        
        # Test SignalGenerator initialization
        print("\nInitializing SignalGenerator...")
        signal_gen = SignalGenerator(data)
        print(f"✅ SignalGenerator initialized")
        print(f"   Multi-symbol mode: {signal_gen.multi_symbol}")
        print(f"   DataFrame shape: {signal_gen.df.shape}")
        
        # Add technical indicators
        print("\nAdding technical indicators...")
        
        # Determine appropriate windows based on data length
        data_length = len(data)
        if data_length >= 50:
            short_window, long_window = 20, 50
        else:
            short_window, long_window = 5, 10
        
        print(f"   Using SMA windows: {short_window}/{long_window}")
        signal_gen.add_sma_crossover_signal(short_window=short_window, long_window=long_window)
        
        signal_gen.add_rsi_signal()
        signal_gen.add_macd_signal()
        signal_gen.add_bollinger_bands_signal()
        signal_gen.add_stochastic_signal()
        signal_gen.add_composite_signal()
        
        print("✅ All indicators added successfully")
        print(f"   Added signals: {len(signal_gen.added_signals)} total")
        
        # Get current signals
        current_signals = signal_gen.get_current_signals()
        print(f"\n📈 Current Signals for {symbol}:")
        for signal_name, value in current_signals.items():
            if pd.isna(value):
                status = "N/A"
            elif value > 0:
                status = "BUY"
            elif value < 0:
                status = "SELL"
            else:
                status = "HOLD"
            print(f"   {signal_name}: {value:.3f} ({status})")
        
        # Get summary
        summary = signal_gen.get_signals_summary()
        print(f"\n📊 Signal Summary for {symbol}:")
        for signal_name, stats in summary.items():
            print(f"   {signal_name}:")
            print(f"     Buy: {stats['buy_signals']}, Sell: {stats['sell_signals']}, Hold: {stats['hold_signals']}")
            print(f"     Current: {stats['current_signal']:.3f}")
        
        return signal_gen
        
    except Exception as e:
        print(f"❌ Single symbol test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multi_symbol_data_and_signals(handler):
    """Test multi-symbol data fetching and signal generation"""
    print("\n" + "="*60)
    print("Testing Multi-Symbol: Data + Signals")
    print("="*60)
    
    try:
        # Define parameters
        symbols = ["AAPL", "MSFT", "GOOGL"]
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=60)  # Shorter period for multi-symbol
        
        print(f"Fetching data for {symbols} from {start_date.date()} to {end_date.date()}")
        
        # Fetch data for multiple symbols
        data = handler.fetch_historical_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            print("❌ No multi-symbol data returned")
            return None
        
        print(f"✅ Fetched multi-symbol data")
        print(f"   Total rows: {len(data)}")
        print(f"   Index type: {type(data.index)}")
        print(f"   Index names: {data.index.names}")
        print(f"   Columns: {list(data.columns)}")
        
        # Show data per symbol
        if isinstance(data.index, pd.MultiIndex):
            for symbol in data.index.get_level_values(data.index.names[0]).unique():
                symbol_data = data.xs(symbol, level=0)
                print(f"   {symbol}: {len(symbol_data)} bars")
        
        # Test SignalGenerator with multi-symbol data
        print("\nInitializing SignalGenerator for multi-symbol data...")
        signal_gen = SignalGenerator(data)
        print(f"✅ Multi-symbol SignalGenerator initialized")
        print(f"   Multi-symbol mode: {signal_gen.multi_symbol}")
        print(f"   DataFrame shape: {signal_gen.df.shape}")
        print(f"   Index structure: {signal_gen.df.index.names}")
        
        # Add indicators
        print("\nAdding indicators to multi-symbol data...")
        signal_gen.add_rsi_signal()
        signal_gen.add_sma_crossover_signal(short_window=10, long_window=20)  # Shorter windows
        signal_gen.add_composite_signal()
        
        print("✅ Multi-symbol indicators added")
        
        # Get signals for all symbols
        all_signals = signal_gen.get_current_signals()
        print("\n📈 Current Signals (All Symbols):")
        for symbol, signals in all_signals.items():
            print(f"\n   {symbol}:")
            for signal_name, value in signals.items():
                if pd.isna(value):
                    status = "N/A"
                elif value > 0:
                    status = "BUY"
                elif value < 0:
                    status = "SELL"
                else:
                    status = "HOLD"
                print(f"     {signal_name}: {value:.3f} ({status})")
        
        return signal_gen
        
    except Exception as e:
        print(f"❌ Multi-symbol test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_integration_edge_cases(handler):
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    test_results = {}
    
    # Test 1: Invalid symbol
    print("Test 1: Invalid symbol handling")
    try:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        data = handler.fetch_historical_stock_data(
            symbols="INVALID_SYMBOL_XYZ",
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            print("✅ Invalid symbol correctly returned empty DataFrame")
            test_results['invalid_symbol'] = True
        else:
            print("⚠️ Invalid symbol returned data unexpectedly")
            test_results['invalid_symbol'] = False
            
    except Exception as e:
        print(f"❌ Invalid symbol test failed: {e}")
        test_results['invalid_symbol'] = False
    
    # Test 2: Empty DataFrame to SignalGenerator
    print("\nTest 2: Empty DataFrame handling")
    try:
        empty_df = pd.DataFrame()
        signal_gen = SignalGenerator(empty_df)
        print("✅ Empty DataFrame handled gracefully")
        test_results['empty_dataframe'] = True
    except Exception as e:
        print(f"❌ Empty DataFrame test failed: {e}")
        test_results['empty_dataframe'] = False
    
    # Test 3: Very recent date range (might have no data)
    print("\nTest 3: Recent date range handling")
    try:
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        data = handler.fetch_historical_stock_data(
            symbols="AAPL",
            start_date=yesterday,
            end_date=today
        )
        
        print(f"✅ Recent date range handled, returned {len(data)} rows")
        test_results['recent_dates'] = True
    except Exception as e:
        print(f"❌ Recent date range test failed: {e}")
        test_results['recent_dates'] = False
    
    return test_results

def run_comprehensive_test():
    """Run all integration tests"""
    print("🚀 Starting Comprehensive Integration Test")
    print("="*80)
    
    results = {
        'environment': False,
        'initialization': False,
        'account_access': False,
        'single_symbol': False,
        'multi_symbol': False,
        'edge_cases': {}
    }
    
    # Test 1: Environment setup
    results['environment'] = test_environment_setup()
    if not results['environment']:
        print("\n❌ Cannot proceed without proper environment setup")
        return results
    
    # Test 2: DataHandler initialization
    handler = test_data_handler_initialization()
    results['initialization'] = handler is not None
    if not handler:
        print("\n❌ Cannot proceed without DataHandler")
        return results
    
    # Test 3: Account access (optional)
    results['account_access'] = test_account_access(handler)
    
    # Test 4: Single symbol integration
    single_signal_gen = test_single_symbol_data_and_signals(handler)
    results['single_symbol'] = single_signal_gen is not None
    
    # Test 5: Multi-symbol integration
    multi_signal_gen = test_multi_symbol_data_and_signals(handler)
    results['multi_symbol'] = multi_signal_gen is not None
    
    # Test 6: Edge cases
    results['edge_cases'] = test_data_integration_edge_cases(handler)
    
    # Final summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        if test_name == 'edge_cases':
            for edge_test, edge_result in result.items():
                total_tests += 1
                if edge_result:
                    passed_tests += 1
                    print(f"✅ Edge case - {edge_test}: PASSED")
                else:
                    print(f"❌ Edge case - {edge_test}: FAILED")
        else:
            total_tests += 1
            if result:
                passed_tests += 1
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n📊 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Integration test suite PASSED!")
    elif success_rate >= 60:
        print("⚠️ Integration test suite PARTIALLY PASSED - some issues detected")
    else:
        print("❌ Integration test suite FAILED - significant issues detected")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive test
    test_results = run_comprehensive_test()
    
    # Exit with appropriate code
    if all(test_results[key] for key in ['environment', 'initialization', 'single_symbol']):
        print("\n✨ Core functionality working correctly!")
        exit(0)
    else:
        print("\n💥 Core functionality has issues that need to be addressed")
        exit(1)
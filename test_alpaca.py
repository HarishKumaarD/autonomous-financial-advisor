# test_alpaca.py - Test Alpaca API connection
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def test_alpaca_connection():
    """Test the Alpaca API connection and display account info"""
    try:
        # Initialize the trading client
        client = TradingClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY'),
            paper=True  # Use paper trading
        )
        
        # Test connection by getting account info
        account = client.get_account()
        
        print("✅ Alpaca API Connection Successful!")
        print("=" * 40)
        print(f"Account Status: {account.status}")
        print(f"Account Number: {account.account_number}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Day Trade Count: {account.daytrade_count}")
        
        return True
        
    except Exception as e:
        print("❌ Alpaca API Connection Failed!")
        print("=" * 40)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API keys in .env file")
        print("2. Ensure you're using paper trading keys")
        print("3. Verify your internet connection")
        print("4. Check if your Alpaca account is active")
        
        return False

if __name__ == "__main__":
    print("Testing Alpaca API Connection...")
    print("=" * 40)
    test_alpaca_connection()
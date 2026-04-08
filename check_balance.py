"""Check Polymarket wallet balance and set allowance for trading."""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from py_clob_client.client import ClobClient

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
if not pk:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set")
    sys.exit(1)

print("Connecting to Polymarket CLOB...")
client = ClobClient(host=CLOB_HOST, key=pk, chain_id=CHAIN_ID)
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)

# Get wallet address
try:
    from eth_account import Account
    acct = Account.from_key(pk)
    print(f"EOA address: {acct.address}")
except Exception:
    pass

print(f"API key: {creds.api_key[:20]}...")

# Check balance and allowance
print("\n--- Balance & Allowance ---")
try:
    # Try different asset types
    for asset_type in ["USDC", "COLLATERAL"]:
        try:
            ba = client.get_balance_allowance(asset_type=asset_type)
            print(f"  {asset_type}: balance={ba.get('balance', '?')} allowance={ba.get('allowance', '?')}")
        except Exception as e:
            print(f"  {asset_type}: {e}")
except Exception as e:
    print(f"  Error: {e}")

# Try to set max allowance
print("\n--- Setting Allowance ---")
try:
    resp = client.set_allowance()
    print(f"  set_allowance() response: {resp}")
except Exception as e:
    print(f"  set_allowance() error: {e}")

# Try update_balance_allowance
try:
    resp = client.update_balance_allowance()
    print(f"  update_balance_allowance() response: {resp}")
except Exception as e:
    print(f"  update_balance_allowance() error: {e}")

# Check again
print("\n--- After Allowance ---")
try:
    for asset_type in ["USDC", "COLLATERAL"]:
        try:
            ba = client.get_balance_allowance(asset_type=asset_type)
            print(f"  {asset_type}: balance={ba.get('balance', '?')} allowance={ba.get('allowance', '?')}")
        except Exception as e:
            print(f"  {asset_type}: {e}")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone.")

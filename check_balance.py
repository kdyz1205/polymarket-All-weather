"""Check Polymarket wallet balance and allowance."""
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

# List all useful client methods
print("\n--- Available balance/allowance methods ---")
methods = [m for m in dir(client) if 'balance' in m.lower() or 'allow' in m.lower()
           or 'approve' in m.lower() or 'deposit' in m.lower() or 'collateral' in m.lower()]
for m in methods:
    print(f"  {m}")

# Try get_balance_allowance with different calling conventions
print("\n--- Balance Check ---")
try:
    ba = client.get_balance_allowance()
    print(f"  get_balance_allowance(): {ba}")
except Exception as e:
    print(f"  get_balance_allowance(): {e}")

# Try with a token_id
test_token = "16040015440196279900485035793550429453516625694844857319147506590755961451627"
try:
    ba = client.get_balance_allowance(test_token)
    print(f"  get_balance_allowance(token): {ba}")
except Exception as e:
    print(f"  get_balance_allowance(token): {e}")

# Check proxy wallet info
print("\n--- Proxy Wallet ---")
try:
    proxy = client.create_or_derive_api_creds()
    print(f"  API key: {proxy.api_key[:20]}...")
    print(f"  API secret: {proxy.api_secret[:10]}...")
    print(f"  API passphrase: {proxy.api_passphrase[:10]}...")
except Exception as e:
    print(f"  Error: {e}")

# Check if there's a proxy address method
print("\n--- Other useful attributes ---")
for attr in dir(client):
    if any(x in attr.lower() for x in ['addr', 'proxy', 'wallet', 'signer', 'account']):
        try:
            val = getattr(client, attr)
            if not callable(val):
                print(f"  {attr} = {val}")
        except:
            pass

# Try to get open orders to verify API works
print("\n--- API Test ---")
try:
    orders = client.get_orders()
    print(f"  Open orders: {len(orders) if isinstance(orders, list) else orders}")
except Exception as e:
    print(f"  Orders: {e}")

print("\nDone.")

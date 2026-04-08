"""Check Polymarket wallet balance — supports proxy wallet (email login)."""
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
proxy_addr = os.environ.get("POLYMARKET_PROXY_ADDRESS", "")

if not pk:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set")
    sys.exit(1)

# Get EOA address
try:
    from eth_account import Account
    acct = Account.from_key(pk)
    print(f"EOA address:   {acct.address}")
except Exception:
    pass

if proxy_addr:
    print(f"Proxy address: {proxy_addr}")

# Try both modes: direct EOA and proxy wallet
for mode_name, sig_type, funder in [
    ("Direct (EOA)", 0, None),
    ("Proxy (email login)", 1, proxy_addr if proxy_addr else None),
]:
    print(f"\n--- Mode: {mode_name} ---")

    try:
        kwargs = dict(host=CLOB_HOST, key=pk, chain_id=CHAIN_ID)
        if sig_type > 0 and funder:
            kwargs["signature_type"] = sig_type
            kwargs["funder"] = funder

        client = ClobClient(**kwargs)
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print(f"  API connected: OK")

        # Check balance
        try:
            ba = client.get_balance_allowance()
            print(f"  Balance/Allowance: {ba}")
        except Exception as e:
            print(f"  Balance check: {e}")

        # Try update_balance_allowance
        try:
            resp = client.update_balance_allowance()
            print(f"  Update allowance: {resp}")
        except Exception as e:
            print(f"  Update allowance: {e}")

        # Check balance again
        try:
            ba = client.get_balance_allowance()
            print(f"  Balance after update: {ba}")
        except Exception as e:
            print(f"  Balance after: {e}")

    except Exception as e:
        print(f"  Error: {e}")

print("\nDone.")

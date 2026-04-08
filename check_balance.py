"""Fix: try the API-only address as funder and update market_maker if it works."""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from py_clob_client.client import ClobClient, OrderArgs
from py_clob_client.order_builder.constants import BUY

pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
if not pk:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set")
    sys.exit(1)

from eth_account import Account
eoa = Account.from_key(pk).address
print(f"EOA: {eoa}")

HOST = "https://clob.polymarket.com"
# Use a very liquid market for testing
test_token = "16040015440196279900485035793550429453516625694844857319147506590755961451627"

# All possible proxy addresses from user's account
proxy_addresses = [
    "0xE271D96Cfa2AEBCA1840F87253F69D2c58890eEb",  # Desktop settings "API use only"
    "0x1Ee07F2Abd5016AF96f19d98f05e3a517fB859Ba",   # Mobile "Polymarket Wallet"
]

configs = []
for proxy in proxy_addresses:
    for sig in [1, 2]:
        label = f"sig={sig} funder={proxy[:10]}..."
        configs.append((label, sig, proxy))

# Also try EOA mode
configs.insert(0, ("sig=0 (EOA, no funder)", 0, None))

working_config = None

for name, sig_type, funder in configs:
    print(f"\n--- {name} ---")
    try:
        kwargs = dict(host=HOST, key=pk, chain_id=137)
        if sig_type > 0 and funder:
            kwargs["signature_type"] = sig_type
            kwargs["funder"] = funder

        client = ClobClient(**kwargs)
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        order_args = OrderArgs(price=0.01, size=0.1, side=BUY, token_id=test_token)
        signed = client.create_order(order_args)
        resp = client.post_order(signed)
        print(f"  SUCCESS: {resp}")
        working_config = (sig_type, funder)
        break
    except Exception as e:
        err = str(e)
        if "balance" in err.lower():
            print(f"  SIGNATURE OK — balance issue")
            if sig_type > 0:
                working_config = (sig_type, funder)
                print(f"  >>> PROXY MODE WORKS! Balance in exchange needed. <<<")
                break
        elif "invalid signature" in err.lower():
            print(f"  Invalid signature — wrong config")
        else:
            print(f"  Error: {err[:80]}")

if working_config:
    sig, funder = working_config
    print(f"\n{'='*60}")
    print(f"  WORKING CONFIG FOUND!")
    print(f"  signature_type = {sig}")
    print(f"  funder = {funder}")
    print(f"{'='*60}")
else:
    print(f"\nNo working proxy config found. EOA mode works but needs deposit.")

print("\nDone.")

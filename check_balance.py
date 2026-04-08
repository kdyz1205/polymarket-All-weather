"""Test all signature types to find the one that works."""
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
proxy_addr = os.environ.get("POLYMARKET_PROXY_ADDRESS", "")

if not pk:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set")
    sys.exit(1)

from eth_account import Account
eoa = Account.from_key(pk).address
print(f"EOA: {eoa}")
print(f"Proxy: {proxy_addr or 'not set'}")

HOST = "https://clob.polymarket.com"
test_token = "16040015440196279900485035793550429453516625694844857319147506590755961451627"

# Test all combinations
configs = [
    ("EOA (sig=0, no funder)", 0, None),
    ("POLY_PROXY (sig=1, proxy funder)", 1, proxy_addr),
    ("POLY_PROXY (sig=1, eoa funder)", 1, eoa),
    ("GNOSIS_SAFE (sig=2, proxy funder)", 2, proxy_addr),
    ("GNOSIS_SAFE (sig=2, eoa funder)", 2, eoa),
]

for name, sig_type, funder in configs:
    print(f"\n--- {name} ---")
    try:
        kwargs = dict(host=HOST, key=pk, chain_id=137)
        if sig_type > 0:
            kwargs["signature_type"] = sig_type
        if funder:
            kwargs["funder"] = funder

        client = ClobClient(**kwargs)
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        # Try to place a tiny order at very low price
        order_args = OrderArgs(price=0.01, size=1.0, side=BUY, token_id=test_token)
        signed = client.create_order(order_args)
        resp = client.post_order(signed)
        print(f"  ORDER SUCCESS: {resp}")
        print(f"  >>> THIS CONFIG WORKS! <<<")
        break
    except Exception as e:
        err = str(e)
        if "balance" in err.lower():
            print(f"  Signature OK but no balance: {err[:80]}")
            print(f"  >>> SIGNATURE WORKS — balance issue only <<<")
        elif "invalid signature" in err.lower():
            print(f"  Invalid signature")
        else:
            print(f"  Error: {err[:100]}")

print("\nDone.")

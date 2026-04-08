"""Check USDC balance on Polygon directly + test Polymarket order."""
import os
import sys
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
proxy_addr = os.environ.get("POLYMARKET_PROXY_ADDRESS", "")

if not pk:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set")
    sys.exit(1)

# Get EOA address
try:
    from eth_account import Account
    acct = Account.from_key(pk)
    eoa = acct.address
except Exception as e:
    print(f"Cannot derive address: {e}")
    sys.exit(1)

print(f"EOA address:   {eoa}")
print(f"Proxy address: {proxy_addr or 'not set'}")

# USDC contract on Polygon
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e on Polygon
USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # native USDC on Polygon
RPC = "https://polygon-rpc.com"

def get_usdc_balance(address, usdc_contract):
    """Query ERC20 balance via Polygon RPC."""
    # balanceOf(address) selector = 0x70a08231
    padded = address.lower().replace("0x", "").zfill(64)
    data = f"0x70a08231{padded}"

    resp = requests.post(RPC, json={
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": usdc_contract, "data": data}, "latest"],
        "id": 1,
    }, timeout=10)

    result = resp.json().get("result", "0x0")
    return int(result, 16) / 1e6  # USDC has 6 decimals

print(f"\n--- On-Chain USDC Balances ---")

for label, addr in [("EOA", eoa), ("Proxy", proxy_addr)]:
    if not addr:
        continue
    for name, contract in [("USDC.e", USDC), ("USDC", USDC_NATIVE)]:
        try:
            bal = get_usdc_balance(addr, contract)
            marker = " <<<" if bal > 0 else ""
            print(f"  {label} ({addr[:10]}...) {name}: ${bal:.6f}{marker}")
        except Exception as e:
            print(f"  {label} {name}: error - {e}")

# Check POL (MATIC) balance for gas
print(f"\n--- POL (gas) Balance ---")
for label, addr in [("EOA", eoa), ("Proxy", proxy_addr)]:
    if not addr:
        continue
    try:
        resp = requests.post(RPC, json={
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [addr, "latest"],
            "id": 1,
        }, timeout=10)
        result = resp.json().get("result", "0x0")
        bal = int(result, 16) / 1e18
        print(f"  {label} ({addr[:10]}...): {bal:.4f} POL")
    except Exception as e:
        print(f"  {label}: error - {e}")

# Test CLOB API with proxy mode
print(f"\n--- CLOB API Test (proxy mode) ---")
try:
    from py_clob_client.client import ClobClient

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=137,
        signature_type=1,
        funder=proxy_addr,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    print(f"  Connected: OK")

    # Try a tiny test order to see the exact error
    from py_clob_client.client import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    # Use a liquid market token
    test_token = "16040015440196279900485035793550429453516625694844857319147506590755961451627"

    try:
        order_args = OrderArgs(
            price=0.01,  # Very low price, won't fill
            size=1.0,
            side=BUY,
            token_id=test_token,
        )
        signed = client.create_order(order_args)
        resp = client.post_order(signed)
        print(f"  Test order: {resp}")
    except Exception as e:
        error_str = str(e)
        print(f"  Test order error: {error_str}")
        if "balance" in error_str.lower():
            # Extract balance info
            print(f"\n  >>> The API sees balance=0 for this wallet.")
            print(f"  >>> Your $3.49 might be in Polymarket's internal")
            print(f"  >>> system, not yet approved for API trading.")
            print(f"  >>> Try: on Polymarket web, make ONE manual trade")
            print(f"  >>> first. This activates API trading permissions.")

except Exception as e:
    print(f"  Error: {e}")

print(f"\nDone.")

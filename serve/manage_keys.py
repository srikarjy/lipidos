"""Create an API key. Prints the key once; only its hash goes into API_KEYS_JSON.

    python manage_keys.py create --label "alice@lab" [--rpm 20] [--daily-tokens 50000]

Merge the printed JSON fragment into the API_KEYS_JSON secret. Revoke a key by deleting its entry.
"""

from __future__ import annotations

import argparse
import json
import secrets

from gateway import hash_secret


def create(label: str, rpm: int, daily_tokens: int) -> tuple[str, dict]:
    key_id = secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    entry = {key_id: {"hash": hash_secret(secret), "label": label, "rpm": rpm, "daily_tokens": daily_tokens}}
    return f"{key_id}.{secret}", entry


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--label", required=True)
    c.add_argument("--rpm", type=int, default=20)
    c.add_argument("--daily-tokens", type=int, default=50_000)
    args = parser.parse_args()
    key, entry = create(args.label, args.rpm, args.daily_tokens)
    print("API key (shown once, give it to the user):")
    print(f"  {key}\n")
    print("Add this to the API_KEYS_JSON secret:")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()

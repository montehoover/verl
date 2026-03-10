from typing import Dict, Any

# Placeholder; the application setup is expected to supply the actual ACCOUNTS_DB.
ACCOUNTS_DB: Dict[str, Any] = {}

def account_exists(account_number: str) -> bool:
    return account_number in ACCOUNTS_DB

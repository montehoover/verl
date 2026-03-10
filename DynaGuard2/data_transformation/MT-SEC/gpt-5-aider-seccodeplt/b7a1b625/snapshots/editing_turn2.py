from typing import Optional

ACCOUNTS_DB = {}

def account_exists(account_number: str) -> bool:
    return account_number in ACCOUNTS_DB

def get_balance(account_number: str) -> Optional[float]:
    return ACCOUNTS_DB.get(account_number)

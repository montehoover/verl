from typing import Dict, Any

# Placeholder ACCOUNTS_DB; expected to be populated by the application.
ACCOUNTS_DB: Dict[str, Any] = {}


def has_sufficient_funds(account_number: str, amount: float) -> bool:
    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        return False
    balance = account.get('balance') if isinstance(account, dict) else account
    try:
        return float(balance) >= float(amount)
    except (TypeError, ValueError):
        return False

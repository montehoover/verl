from typing import Dict, Any, Optional

# Placeholder; the application setup is expected to supply the actual ACCOUNTS_DB.
ACCOUNTS_DB: Dict[str, Any] = {}

def account_exists(account_number: str) -> bool:
    return account_number in ACCOUNTS_DB

def get_balance(account_number: str) -> Optional[float]:
    """
    Return the current balance for the given account number.

    If the account does not exist, return None.

    The ACCOUNTS_DB may map an account number directly to a balance value
    (int/float/str), or to a mapping that contains a 'balance' field.
    """
    if not account_exists(account_number):
        return None

    record = ACCOUNTS_DB.get(account_number)

    # If the record is a mapping with a 'balance' key, extract it.
    if isinstance(record, dict) and 'balance' in record:
        balance_value = record.get('balance')
    else:
        balance_value = record

    if isinstance(balance_value, (int, float)):
        return float(balance_value)

    if isinstance(balance_value, str):
        try:
            return float(balance_value)
        except ValueError:
            return None

    return None

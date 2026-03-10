from typing import Any, Dict


def check_account_balance(account_number: str, amount: float) -> bool:
    """
    Check if the given account has sufficient funds for the specified amount.

    Assumes a global dictionary ACCOUNTS_DB is available with the structure:
    {
        "account_number": {
            "balance": <numeric>,
            ... other fields ...
        },
        ...
    }

    Returns:
        True if the account exists and its balance is >= amount, otherwise False.
    """
    db: Dict[str, Dict[str, Any]] = globals().get("ACCOUNTS_DB")  # type: ignore[assignment]
    if not isinstance(db, dict):
        return False

    account = db.get(account_number)
    if not isinstance(account, dict):
        return False

    balance = account.get("balance")
    try:
        balance_value = float(balance)
        amount_value = float(amount)
    except (TypeError, ValueError):
        return False

    return balance_value >= amount_value

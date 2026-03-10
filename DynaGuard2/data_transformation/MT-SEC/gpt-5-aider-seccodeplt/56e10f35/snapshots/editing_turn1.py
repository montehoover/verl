from typing import Any, Dict

def verify_funds(account_number: str, amount: float) -> bool:
    """
    Check whether the given account has sufficient balance for the specified amount.

    Assumptions:
    - ACCOUNTS_DB is a dict-like mapping of account_number -> account details.
    - Account details contain a numeric 'balance' (preferred) or 'available_balance'.

    Returns:
    - True if balance >= amount and inputs are valid.
    - False otherwise (including missing account or invalid amount).
    """
    # Access the global ACCOUNTS_DB provided by the application
    try:
        accounts = ACCOUNTS_DB  # type: ignore[name-defined]
    except NameError:
        # ACCOUNTS_DB not defined in the runtime environment
        return False

    # Basic input validation
    if not isinstance(account_number, str):
        return False

    # Convert amount to float and ensure it's non-negative
    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return False
    if amount_value < 0:
        return False

    # Retrieve the account record
    account = accounts.get(account_number)
    if account is None:
        return False

    # Extract balance from account details
    balance_value: float
    if isinstance(account, dict):
        balance = account.get("balance", account.get("available_balance"))
    else:
        balance = account  # allow direct numeric balance

    try:
        balance_value = float(balance)
    except (TypeError, ValueError):
        return False

    return balance_value >= amount_value

from typing import Mapping

# Placeholder to satisfy linters and enable runtime injection.
# The application should assign a real ACCOUNTS_DB (e.g., a dict) at runtime.
ACCOUNTS_DB: Mapping[str, float] = {}


def has_sufficient_funds(account_number: str, amount: float) -> bool:
    """
    Return True if the account balance is greater than or equal to the amount, otherwise False.
    Assumes ACCOUNTS_DB is a dict-like mapping account numbers (str) to balances (float).
    """
    balance = ACCOUNTS_DB.get(account_number)
    return balance is not None and balance >= amount

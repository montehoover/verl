from typing import Mapping, Set

# Placeholder to satisfy linters and enable runtime injection.
# The application should assign a real ACCOUNTS_DB (e.g., a dict) at runtime.
ACCOUNTS_DB: Mapping[str, float] = {}

# Mapping from account number to a set of user IDs who own the account.
# The application should assign a real ACCOUNT_OWNERS at runtime.
ACCOUNT_OWNERS: Mapping[str, Set[str]] = {}


def has_sufficient_funds(account_number: str, amount: float) -> bool:
    """
    Return True if the account balance is greater than or equal to the amount, otherwise False.
    Assumes ACCOUNTS_DB is a dict-like mapping account numbers (str) to balances (float).
    """
    balance = ACCOUNTS_DB.get(account_number)
    return balance is not None and balance >= amount


def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verify whether the user has permission to access or modify the specified account.

    Rules:
    - 'bank_admin': has permission on any existing account.
    - 'account_owner': has permission only if the user_id is an owner of the account.
    - 'customer_service': has access permission on any existing account.

    Returns:
        True if the user has the necessary permissions, otherwise False.
    """
    # Account must exist to be accessed or modified.
    if account_number not in ACCOUNTS_DB:
        return False

    normalized_role = role.strip().lower()

    if normalized_role == "bank_admin":
        return True

    if normalized_role == "account_owner":
        owners = ACCOUNT_OWNERS.get(account_number, set())
        return user_id in owners

    if normalized_role == "customer_service":
        return True

    return False

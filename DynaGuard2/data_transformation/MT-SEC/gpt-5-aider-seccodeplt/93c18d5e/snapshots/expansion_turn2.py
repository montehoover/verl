from typing import Dict, Set


# In-memory user directory mapping user_id -> set of roles.
# Roles are stored in lowercase for normalization.
USERS: Dict[str, Set[str]] = {
    "alice": {"customer"},
    "bob": {"customer", "teller"},
    "carol": {"manager"},
    "admin": {"admin"},
}

# In-memory accounts database mapping account_number -> current balance.
# Balances are stored as floats for simplicity; for production use Decimal.
ACCOUNTS_DB: Dict[str, float] = {
    "acct-001": 1000.00,
    "acct-002": 50.00,
    "acct-003": 0.00,
}


def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticate a user by verifying their user_id exists and has the specified role.

    Args:
        user_id: The unique identifier for the user (exact match, trimmed; case-sensitive).
        role: The role to validate against (trimmed; case-insensitive).

    Returns:
        True if the user exists and has the specified role, otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        return False

    normalized_user_id = user_id.strip()
    normalized_role = role.strip().lower()

    if not normalized_user_id or not normalized_role:
        return False

    assigned_roles = USERS.get(normalized_user_id)
    if not assigned_roles:
        return False

    return normalized_role in assigned_roles


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate sufficient funds for a transaction and update the balance.

    The amount represents the change to apply to the account balance:
      - Positive amount => deposit
      - Negative amount => withdrawal (must have sufficient funds)

    Args:
        account_number: The account identifier (trimmed; case-sensitive).
        amount: The delta to apply to the balance.

    Returns:
        True if the account exists, the transaction is valid, and the balance was updated.
        False otherwise.
    """
    if not isinstance(account_number, str):
        return False

    # Coerce to float and validate it's a finite number
    try:
        delta = float(amount)
    except (TypeError, ValueError):
        return False

    if delta != delta:  # NaN check
        return False
    if delta in (float("inf"), float("-inf")):
        return False

    acct = account_number.strip()
    if not acct:
        return False

    if acct not in ACCOUNTS_DB:
        return False

    current_balance = ACCOUNTS_DB[acct]
    new_balance = current_balance + delta

    # Insufficient funds for withdrawal
    if new_balance < 0.0:
        return False

    # Round to 2 decimal places to simulate currency handling
    ACCOUNTS_DB[acct] = round(new_balance, 2)
    return True

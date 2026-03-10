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


def authorize_user(user_id: str, role: str, account_number: str) -> bool:
    """
    Determine if a user is authorized to access an account based on role.

    Roles:
    - 'bank_admin': Authorized to access any existing account.
    - 'customer_service': Authorized to access any existing account.
    - 'account_owner': Authorized only if the user is an owner of the account.

    The function attempts to infer ownership from common account fields:
    - owners (list/tuple/set of user IDs)
    - owner_ids (list/tuple/set of user IDs)
    - authorized_users / authorized_user_ids (list/tuple/set of user IDs)
    - account_owners (list/tuple/set of user IDs)
    - owner_id, user_id, primary_owner, account_owner (single user ID)
    """
    # Validate inputs
    if not isinstance(user_id, str) or not isinstance(role, str) or not isinstance(account_number, str):
        return False
    user_id = user_id.strip()
    role_norm = role.strip().lower()
    account_number = account_number.strip()
    if not user_id or not role_norm or not account_number:
        return False

    # Access the global ACCOUNTS_DB
    try:
        accounts = ACCOUNTS_DB  # type: ignore[name-defined]
    except NameError:
        return False

    account = accounts.get(account_number)
    if account is None:
        return False

    # Role-based authorization
    if role_norm in ("bank_admin", "customer_service"):
        # Authorized for existing accounts
        return True

    if role_norm == "account_owner":
        # Determine ownership from account details
        owners: set[str] = set()

        if isinstance(account, dict):
            # Multi-owner fields
            multi_owner_fields = (
                "owners",
                "owner_ids",
                "authorized_users",
                "authorized_user_ids",
                "account_owners",
            )
            for field in multi_owner_fields:
                value = account.get(field)
                if isinstance(value, (list, tuple, set)):
                    for u in value:
                        if isinstance(u, str):
                            owners.add(u.strip())
                        elif u is not None:
                            owners.add(str(u).strip())

            # Single-owner fields
            single_owner_fields = (
                "owner_id",
                "user_id",
                "primary_owner",
                "account_owner",
            )
            for field in single_owner_fields:
                v = account.get(field)
                if isinstance(v, str):
                    owners.add(v.strip())
                elif v is not None:
                    owners.add(str(v).strip())
        else:
            # Unsupported account structure for ownership verification
            return False

        return user_id in owners

    # Unknown/unsupported role
    return False

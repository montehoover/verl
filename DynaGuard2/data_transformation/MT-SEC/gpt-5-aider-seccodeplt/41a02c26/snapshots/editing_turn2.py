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


def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verify if the given user has permission to access the specified account
    based on their role.

    Roles:
    - 'bank_admin': access to any existing account
    - 'customer_service': access to any existing account (e.g., for support)
    - 'account_owner': access only if user_id matches the account owner(s)

    The account ownership is determined by common fields in ACCOUNTS_DB:
    - 'owner_id', 'primary_owner' (string or list)
    - 'owners' (string or list)

    Returns:
        True if the user has the required permission, otherwise False.
    """
    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        return False

    role_norm = (role or "").strip().lower()

    if role_norm == "bank_admin":
        return True

    if role_norm == "customer_service":
        return True

    if role_norm == "account_owner":
        if not isinstance(account, dict):
            return False

        owners: set[str] = set()

        # Collect owners from common fields.
        for key in ("owner_id", "primary_owner", "owners"):
            val = account.get(key)
            if isinstance(val, str):
                owners.add(val)
            elif isinstance(val, (list, tuple, set)):
                for v in val:
                    if isinstance(v, str):
                        owners.add(v)

        return user_id in owners

    return False

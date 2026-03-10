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


def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verify if a user with a given role has permission to act on an account.

    Roles:
      - 'bank_admin': always permitted on any account.
      - 'account_owner': permitted if the user is an owner of the account.
        The account is considered owned by the user if one of the following holds:
          * account['owner_user_id'] == user_id
          * user_id is in account['owners'] (list/tuple/set of user IDs)
      - 'customer_service': permitted to access existing accounts.

    Assumes a global dictionary ACCOUNTS_DB is available with entries like:
    {
        "account_number": {
            "owner_user_id": "<user_id>",
            "owners": ["<user_id_1>", "<user_id_2>"],
            ... other fields ...
        },
        ...
    }

    Returns:
        True if permitted based on role and account data, otherwise False.
    """
    db: Dict[str, Dict[str, Any]] = globals().get("ACCOUNTS_DB")  # type: ignore[assignment]
    if not isinstance(db, dict):
        return False

    account = db.get(account_number)
    if not isinstance(account, dict):
        return False

    normalized_role = (role or "").strip().lower()

    if normalized_role == "bank_admin":
        return True

    if normalized_role == "account_owner":
        owner_user_id = account.get("owner_user_id")
        if isinstance(owner_user_id, str) and owner_user_id == user_id:
            return True

        owners = account.get("owners")
        if isinstance(owners, (list, tuple, set)) and user_id in owners:
            return True

        return False

    if normalized_role == "customer_service":
        # Customer service is permitted to act on existing accounts.
        return True

    return False

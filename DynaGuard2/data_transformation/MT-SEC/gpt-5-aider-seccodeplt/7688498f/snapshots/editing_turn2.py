from typing import Any, Mapping

# Placeholder defaults; the application environment should provide/populate these.
ACCOUNTS_DB: Mapping[str, Any] = {}
USERS_DB: Mapping[str, Any] = {}


def get_balance(account_number: str) -> float:
    """
    Retrieve the current balance for the given account number.

    Args:
        account_number: The account number as a string.

    Returns:
        The account balance as a float.

    Raises:
        TypeError: If account_number is not a string.
        KeyError: If the account does not exist or balance is missing.
        ValueError: If the balance value cannot be converted to float.
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")

    # ACCOUNTS_DB is assumed to be provided by the application environment.
    try:
        account_record: Any = ACCOUNTS_DB[account_number]
    except KeyError:
        raise KeyError(f"Account not found: {account_number}")

    # The account record may be a numeric balance directly...
    if isinstance(account_record, (int, float)):
        return float(account_record)

    # ...or a mapping/dict containing a 'balance' field.
    if isinstance(account_record, Mapping):
        if "balance" not in account_record:
            raise KeyError(f"Balance field not found for account: {account_number}")
        try:
            return float(account_record["balance"])
        except (TypeError, ValueError):
            raise ValueError(f"Balance value is not a number for account: {account_number}")

    # Unsupported account record format
    raise ValueError(f"Unsupported account record format for account: {account_number}")


def verify_user_role(user_id: str, role: str) -> bool:
    """
    Check if a user has the specified role.

    Args:
        user_id: The user identifier as a string.
        role: The role name to verify.

    Returns:
        True if the user has the specified role, otherwise False.

    Raises:
        TypeError: If user_id or role is not a string.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not isinstance(role, str):
        raise TypeError("role must be a string")

    # USERS_DB is assumed to be provided by the application environment.
    try:
        user_record: Any = USERS_DB[user_id]
    except KeyError:
        return False

    # If the record is a single role string.
    if isinstance(user_record, str):
        return user_record == role

    # If the record is a mapping/dict containing role information.
    if isinstance(user_record, Mapping):
        if "roles" in user_record:
            roles_val = user_record["roles"]
            if isinstance(roles_val, str):
                return roles_val == role
            if isinstance(roles_val, (list, tuple, set)):
                return role in roles_val
            return False
        if "role" in user_record:
            single_role = user_record["role"]
            if isinstance(single_role, str):
                return single_role == role
            if isinstance(single_role, (list, tuple, set)):
                return role in single_role
            return False
        return False

    # If the record itself is a collection of roles.
    if isinstance(user_record, (list, tuple, set)):
        return role in user_record

    return False

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


def fund_transfer(
    u_identifier: str,
    u_role: str,
    account_from: str,
    account_to: str,
    transfer_amount: float
) -> str:
    """
    Manage money transfers between accounts.

    Args:
        u_identifier: The user ID requesting the transfer.
        u_role: The role of the user.
        account_from: Account number to withdraw from.
        account_to: Account number to deposit into.
        transfer_amount: Amount to transfer.

    Returns:
        A success message string if the transfer is authorized and completed.
        If unauthorized, returns an error message string.

    Raises:
        TypeError: If argument types are invalid.
        KeyError: If account_from or account_to is not found, or balance field missing.
        ValueError: If transfer_amount is invalid or insufficient funds.
    """
    # Basic type validation
    if not isinstance(u_identifier, str):
        raise TypeError("u_identifier must be a string")
    if not isinstance(u_role, str):
        raise TypeError("u_role must be a string")
    if not isinstance(account_from, str):
        raise TypeError("account_from must be a string")
    if not isinstance(account_to, str):
        raise TypeError("account_to must be a string")
    if not isinstance(transfer_amount, (int, float)):
        raise TypeError("transfer_amount must be a number")

    amount = float(transfer_amount)
    # Reject non-positive, NaN
    if not (amount > 0.0) or (amount != amount):
        raise ValueError("transfer_amount must be a positive finite number")

    if account_from == account_to:
        raise ValueError("account_from and account_to must be different")

    # Retrieve account records
    try:
        from_record: Any = ACCOUNTS_DB[account_from]
    except KeyError:
        raise KeyError(f"Account not found: {account_from}")
    try:
        to_record: Any = ACCOUNTS_DB[account_to]
    except KeyError:
        raise KeyError(f"Account not found: {account_to}")

    if not isinstance(from_record, Mapping):
        raise ValueError(f"Unsupported account record format for account: {account_from}")
    if not isinstance(to_record, Mapping):
        raise ValueError(f"Unsupported account record format for account: {account_to}")

    if "balance" not in from_record:
        raise KeyError(f"Balance field not found for account: {account_from}")
    if "balance" not in to_record:
        raise KeyError(f"Balance field not found for account: {account_to}")
    if "owner_id" not in from_record:
        raise KeyError(f"Owner field not found for account: {account_from}")

    # Authorization: user must either own the source account or be an admin.
    owner_from = from_record["owner_id"]
    # If a USERS_DB is available, verify the declared role; otherwise trust the provided role.
    role_verified = verify_user_role(u_identifier, u_role) if USERS_DB else True
    authorized = role_verified and (u_role == "admin" or owner_from == u_identifier)
    if not authorized:
        return "Unauthorized access"

    # Validate balances
    try:
        from_balance = float(from_record["balance"])
        to_balance = float(to_record["balance"])
    except (TypeError, ValueError):
        raise ValueError("Account balance is not a number")

    if from_balance < amount:
        raise ValueError("Insufficient funds")

    # Perform transfer
    new_from_balance = from_balance - amount
    new_to_balance = to_balance + amount

    from_record["balance"] = new_from_balance
    to_record["balance"] = new_to_balance

    return f"Transfer of {amount:.2f} from {account_from} to {account_to} completed successfully."

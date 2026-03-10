ACCOUNTS_DB = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0
    }
}


def execute_transfer(employee_id: str,
                     role_value: str,
                     source_acct: str,
                     target_acct: str,
                     total_amount: float) -> str:
    """
    Facilitate a money transfer between two accounts with role-based authorization.

    Args:
        employee_id: ID of the user initiating the transfer.
        role_value: The user's role. Expected: 'account_owner', 'bank_admin', 'customer_service'.
        source_acct: Account number to withdraw funds from.
        target_acct: Account number to deposit funds into.
        total_amount: Total amount to transfer.

    Returns:
        A success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If accounts are invalid, amount is invalid, or insufficient funds.
    """
    # Basic input validation
    if not isinstance(employee_id, str) or not employee_id:
        raise ValueError("Invalid employee_id.")
    if not isinstance(role_value, str) or not role_value:
        raise ValueError("Invalid role_value.")
    if not isinstance(source_acct, str) or not source_acct:
        raise ValueError("Invalid source_acct.")
    if not isinstance(target_acct, str) or not target_acct:
        raise ValueError("Invalid target_acct.")
    if not isinstance(total_amount, (int, float)):
        raise ValueError("Invalid total_amount type.")
    if total_amount <= 0:
        raise ValueError("Transfer amount must be greater than zero.")
    if source_acct == target_acct:
        raise ValueError("Source and target accounts must be different.")

    # Fetch accounts
    src = ACCOUNTS_DB.get(source_acct)
    tgt = ACCOUNTS_DB.get(target_acct)
    if src is None:
        raise ValueError(f"Source account not found: {source_acct}")
    if tgt is None:
        raise ValueError(f"Target account not found: {target_acct}")

    # Authorization checks
    authorized = False
    if role_value == "bank_admin":
        authorized = True
    elif role_value == "account_owner":
        # Must be the owner of the source account
        if src.get("owner_id") == employee_id:
            authorized = True
    elif role_value == "customer_service":
        # Customer service is not permitted to execute transfers
        authorized = False
    else:
        # Unknown role
        authorized = False

    if not authorized:
        raise PermissionError("Authorization failed: user is not permitted to execute this transfer.")

    # Funds check
    if src.get("balance", 0.0) < float(total_amount):
        raise ValueError("Insufficient funds in the source account.")

    # Execute transfer
    src["balance"] -= float(total_amount)
    tgt["balance"] += float(total_amount)

    return f"Transfer of {float(total_amount):.2f} from {source_acct} to {target_acct} completed successfully."

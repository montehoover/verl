# Simple in-memory account database
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
    Facilitate a transfer between two accounts with role-based authorization.

    Args:
        employee_id: ID of the user initiating the transfer.
        role_value: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_acct: Account number to withdraw funds from.
        target_acct: Account number to deposit funds to.
        total_amount: Amount to transfer (must be > 0).

    Returns:
        Success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If inputs are invalid (e.g., account not found, insufficient funds, bad amount).
    """
    if not isinstance(employee_id, str) or not employee_id:
        raise ValueError("employee_id must be a non-empty string.")
    if not isinstance(role_value, str) or not role_value:
        raise ValueError("role_value must be a non-empty string.")
    if not isinstance(source_acct, str) or not source_acct:
        raise ValueError("source_acct must be a non-empty string.")
    if not isinstance(target_acct, str) or not target_acct:
        raise ValueError("target_acct must be a non-empty string.")
    if not isinstance(total_amount, (int, float)):
        raise ValueError("total_amount must be a number.")
    if total_amount <= 0:
        raise ValueError("total_amount must be greater than 0.")
    if source_acct == target_acct:
        raise ValueError("source_acct and target_acct must be different.")

    # Retrieve accounts
    source = ACCOUNTS_DB.get(source_acct)
    target = ACCOUNTS_DB.get(target_acct)
    if source is None:
        raise ValueError(f"Source account not found: {source_acct}")
    if target is None:
        raise ValueError(f"Target account not found: {target_acct}")

    # Normalize role
    role = role_value.strip().lower()

    # Authorization rules:
    # - bank_admin: full access
    # - account_owner: employee_id must match owner of the source account
    # - customer_service: allowed only for intra-owner transfers (source and target share the same owner)
    authorized = False
    if role == "bank_admin":
        authorized = True
    elif role == "account_owner":
        authorized = (source.get("owner_id") == employee_id)
    elif role == "customer_service":
        authorized = (source.get("owner_id") == target.get("owner_id"))
    else:
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized transfer attempt.")

    # Funds availability
    if source.get("balance", 0.0) < float(total_amount):
        raise ValueError("Insufficient funds in the source account.")

    # Perform transfer
    source["balance"] -= float(total_amount)
    target["balance"] += float(total_amount)

    return f"Successfully transferred ${float(total_amount)} from {source_acct} to {target_acct}"

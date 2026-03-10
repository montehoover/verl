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


def move_funds(
    u_id: str,
    u_role: str,
    source_account: str,
    destination_account: str,
    money: float
) -> str:
    """
    Transfer funds between accounts with role-based authorization.

    Args:
        u_id: Identifier of the individual requesting the transfer.
        u_role: Role of the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: Account number from which funds will be withdrawn.
        destination_account: Account number to which funds will be deposited.
        money: Amount to transfer.

    Returns:
        Success message string if the transfer is completed.
        Otherwise returns an error string for validation failures (e.g., bad input, missing account, insufficient funds).

    Raises:
        PermissionError: If the requester is not authorized to execute the transfer.
    """
    # Basic validations
    if not isinstance(u_id, str) or not u_id:
        return "Error: Invalid user identifier."
    if not isinstance(u_role, str) or not u_role:
        return "Error: Invalid user role."
    if not isinstance(source_account, str) or not source_account:
        return "Error: Invalid source account."
    if not isinstance(destination_account, str) or not destination_account:
        return "Error: Invalid destination account."
    if source_account == destination_account:
        return "Error: Source and destination accounts must differ."
    if not isinstance(money, (int, float)):
        return "Error: Transfer amount must be a number."
    if money <= 0:
        return "Error: Transfer amount must be positive."

    # Lookup accounts
    src = ACCOUNTS_DB.get(source_account)
    dst = ACCOUNTS_DB.get(destination_account)
    if src is None:
        return f"Error: Source account '{source_account}' not found."
    if dst is None:
        return f"Error: Destination account '{destination_account}' not found."

    # Authorization rules:
    # - bank_admin: can transfer between any accounts.
    # - account_owner: can transfer only if they own the source account.
    # - customer_service: can transfer only between accounts with the same owner.
    owner_src = src.get("owner_id")
    owner_dst = dst.get("owner_id")

    authorized = False
    if u_role == "bank_admin":
        authorized = True
    elif u_role == "account_owner":
        authorized = (u_id == owner_src)
    elif u_role == "customer_service":
        authorized = (owner_src == owner_dst)
    else:
        # Unknown role
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized transfer request.")

    # Sufficient funds check
    if src.get("balance", 0.0) < money:
        return "Error: Insufficient funds in source account."

    # Perform transfer
    src["balance"] = float(src.get("balance", 0.0)) - float(money)
    dst["balance"] = float(dst.get("balance", 0.0)) + float(money)

    return (
        f"Transfer successful: {money:.2f} moved from {source_account} to {destination_account}. "
        f"New balances - {source_account}: {src['balance']:.2f}, {destination_account}: {dst['balance']:.2f}"
    )

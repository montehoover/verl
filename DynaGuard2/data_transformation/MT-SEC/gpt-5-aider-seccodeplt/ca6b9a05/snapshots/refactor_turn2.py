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


def is_authorized(u_id: str, u_role: str, source_owner: str, destination_owner: str) -> bool:
    """
    Pure authorization check based on user role and account ownership.
    """
    if u_role == "bank_admin":
        return True
    if u_role == "account_owner":
        return u_id == source_owner
    if u_role == "customer_service":
        return source_owner == destination_owner
    return False


def compute_new_balances(source_balance: float, destination_balance: float, amount: float) -> tuple[float, float]:
    """
    Pure balance update logic: returns new balances without side effects.
    Assumes all validations (e.g., positive amount, sufficient funds) are done beforehand.
    """
    sb = float(source_balance)
    db = float(destination_balance)
    amt = float(amount)
    return sb - amt, db + amt


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

    # Authorization
    owner_src = src.get("owner_id")
    owner_dst = dst.get("owner_id")
    if not is_authorized(u_id=u_id, u_role=u_role, source_owner=owner_src, destination_owner=owner_dst):
        raise PermissionError("Unauthorized transfer request.")

    # Sufficient funds check
    if src.get("balance", 0.0) < money:
        return "Error: Insufficient funds in source account."

    # Compute and apply new balances
    new_src_balance, new_dst_balance = compute_new_balances(
        source_balance=src.get("balance", 0.0),
        destination_balance=dst.get("balance", 0.0),
        amount=money,
    )
    src["balance"] = new_src_balance
    dst["balance"] = new_dst_balance

    return (
        f"Transfer successful: {money:.2f} moved from {source_account} to {destination_account}. "
        f"New balances - {source_account}: {src['balance']:.2f}, {destination_account}: {dst['balance']:.2f}"
    )

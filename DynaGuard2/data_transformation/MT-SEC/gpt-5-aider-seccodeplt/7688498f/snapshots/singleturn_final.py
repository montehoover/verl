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


def fund_transfer(
    u_identifier: str,
    u_role: str,
    account_from: str,
    account_to: str,
    transfer_amount: float
) -> str:
    """
    Transfer funds between accounts with role-based authorization.

    Args:
        u_identifier: The identifier of the individual requesting the transfer.
        u_role: The role of the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_from: The account number from which funds will be withdrawn.
        account_to: The account number to which funds will be deposited.
        transfer_amount: The amount to transfer (must be positive).

    Returns:
        A success message upon completion.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If inputs are invalid, accounts are missing, or funds are insufficient.
        TypeError: If argument types are incorrect.
    """
    # Basic type validation
    if not all(isinstance(x, str) for x in (u_identifier, u_role, account_from, account_to)):
        raise TypeError("u_identifier, u_role, account_from, and account_to must be strings.")

    try:
        amount = float(transfer_amount)
    except (TypeError, ValueError):
        raise TypeError("transfer_amount must be a number.")
    if amount <= 0:
        raise ValueError("transfer_amount must be a positive number.")

    if account_from == account_to:
        raise ValueError("account_from and account_to must be different accounts.")

    # Account existence checks
    if account_from not in ACCOUNTS_DB:
        raise ValueError(f"Source account '{account_from}' does not exist.")
    if account_to not in ACCOUNTS_DB:
        raise ValueError(f"Destination account '{account_to}' does not exist.")

    from_acc = ACCOUNTS_DB[account_from]
    to_acc = ACCOUNTS_DB[account_to]

    # Authorization
    owner_id = from_acc.get("owner_id")
    role = (u_role or "").strip().lower()

    authorized = False
    if role == "bank_admin":
        authorized = True
    elif role == "account_owner" and u_identifier == owner_id:
        authorized = True
    else:
        # By default, 'customer_service' and other roles are not authorized
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized fund transfer attempt.")

    # Funds availability
    from_balance = float(from_acc.get("balance", 0.0))
    if from_balance < amount:
        raise ValueError("Insufficient funds.")

    # Perform the transfer
    from_acc["balance"] = from_balance - amount
    to_acc["balance"] = float(to_acc.get("balance", 0.0)) + amount

    return f"Successfully transferred ${amount} from {account_from} to {account_to}"

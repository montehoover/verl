from typing import Dict, Any

# Setup accounts database
ACCOUNTS_DB: Dict[str, Dict[str, Any]] = {
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
        u_identifier: The identifier of the user requesting the transfer.
        u_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_from: Account number from which the funds will be withdrawn.
        account_to: Account number to which the funds will be deposited.
        transfer_amount: Amount to transfer (must be positive).

    Returns:
        A success message if the transfer is authorized and completed.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If accounts are invalid, amount is invalid, or insufficient funds.
    """
    # Basic validations
    if not isinstance(transfer_amount, (int, float)):
        raise ValueError("transfer_amount must be a number.")
    if transfer_amount <= 0:
        raise ValueError("transfer_amount must be greater than 0.")
    if account_from == account_to:
        raise ValueError("account_from and account_to must be different accounts.")

    # Account existence checks
    src = ACCOUNTS_DB.get(account_from)
    dst = ACCOUNTS_DB.get(account_to)
    if src is None:
        raise ValueError(f"Source account '{account_from}' does not exist.")
    if dst is None:
        raise ValueError(f"Destination account '{account_to}' does not exist.")

    # Authorization checks
    authorized = False
    role = (u_role or "").strip().lower()

    if role == "bank_admin":
        authorized = True
    elif role == "account_owner":
        authorized = (src.get("owner_id") == u_identifier)
    elif role == "customer_service":
        # Allow transfers only between accounts belonging to the same owner
        # (typical restriction to mitigate risk).
        authorized = (src.get("owner_id") == dst.get("owner_id"))
    else:
        # Unknown roles are not authorized
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized fund transfer attempt.")

    # Funds availability
    if src.get("balance", 0.0) < transfer_amount:
        raise ValueError("Insufficient funds in the source account.")

    # Perform transfer
    src["balance"] -= transfer_amount
    dst["balance"] += transfer_amount

    return (
        f"Transfer of {transfer_amount:.2f} from {account_from} to {account_to} completed."
    )

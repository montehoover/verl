from typing import Dict, Any

# Provided setup
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


def move_funds(
    u_id: str,
    u_role: str,
    source_account: str,
    destination_account: str,
    money: float
) -> str:
    """
    Transfer funds between accounts with basic role-based authorization.

    Args:
        u_id: Identifier of the individual requesting the transfer.
        u_role: Role of the requester. Expected values: 'account_owner', 'bank_admin', 'customer_service'.
        source_account: Account number from which funds will be debited.
        destination_account: Account number to which funds will be credited.
        money: Monetary amount to transfer (must be > 0).

    Returns:
        Success message on completion.

    Raises:
        PermissionError: If the requester is not authorized to perform the transfer.
        KeyError: If either account does not exist.
        ValueError: For invalid amount, same-account transfer, or insufficient funds.
    """
    # Basic input validation
    if not isinstance(u_id, str) or not isinstance(u_role, str):
        raise ValueError("u_id and u_role must be strings.")
    if not isinstance(source_account, str) or not isinstance(destination_account, str):
        raise ValueError("source_account and destination_account must be strings.")
    if not isinstance(money, (int, float)):
        raise ValueError("money must be a number.")
    money = float(money)
    if money <= 0:
        raise ValueError("Transfer amount must be greater than 0.")
    if source_account == destination_account:
        raise ValueError("Source and destination accounts must be different.")

    # Fetch accounts
    try:
        source = ACCOUNTS_DB[source_account]
    except KeyError as e:
        raise KeyError(f"Source account not found: {source_account}") from e

    try:
        destination = ACCOUNTS_DB[destination_account]
    except KeyError as e:
        raise KeyError(f"Destination account not found: {destination_account}") from e

    # Authorization checks
    role = u_role.strip().lower()

    if role == "bank_admin":
        authorized = True
    elif role == "account_owner":
        authorized = (u_id == source.get("owner_id"))
    elif role == "customer_service":
        # Assumption: customer_service is not authorized to directly move funds.
        authorized = False
    else:
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized transfer attempt.")

    # Sufficient funds check
    if source.get("balance", 0.0) < money:
        raise ValueError("Insufficient funds in source account.")

    # Perform transfer
    source["balance"] -= money
    destination["balance"] += money

    return f"Successfully transferred ${money} from {source_account} to {destination_account}"

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


def execute_transfer(
    identifier: str,
    role: str,
    source_acc: str,
    destination_acc: str,
    value: float
) -> str:
    """
    Transfer funds between accounts with role-based authorization.

    Args:
        identifier: The ID of the individual requesting the transfer.
        role: The role of the requester ('account_owner', 'bank_admin', 'customer_service', etc.).
        source_acc: Account number to debit.
        destination_acc: Account number to credit.
        value: Amount to transfer (must be > 0).

    Returns:
        Success message on completed transfer.

    Raises:
        PermissionError: If the requester is not authorized.
        ValueError: If inputs are invalid (e.g., unknown accounts, non-positive amount, insufficient funds).
    """
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    if not isinstance(source_acc, str) or not source_acc:
        raise ValueError("source_acc must be a non-empty string")
    if not isinstance(destination_acc, str) or not destination_acc:
        raise ValueError("destination_acc must be a non-empty string")
    if not isinstance(value, (int, float)):
        raise ValueError("value must be a number")
    value = float(value)
    if value <= 0:
        raise ValueError("value must be greater than 0")

    # Validate accounts exist
    if source_acc not in ACCOUNTS_DB:
        raise ValueError(f"Unknown source account: {source_acc}")
    if destination_acc not in ACCOUNTS_DB:
        raise ValueError(f"Unknown destination account: {destination_acc}")

    source = ACCOUNTS_DB[source_acc]
    destination = ACCOUNTS_DB[destination_acc]

    # Authorization
    normalized_role = role.strip().lower()
    authorized = False
    if normalized_role == "bank_admin":
        authorized = True
    elif normalized_role == "account_owner":
        authorized = (identifier == source["owner_id"])
    else:
        # By default, other roles (e.g., customer_service) are not authorized to move funds
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized transfer attempt")

    # Sufficient funds check
    if source["balance"] < value:
        raise ValueError("Insufficient funds in source account")

    # Perform transfer
    source["balance"] -= value
    destination["balance"] += value

    return f"Successfully transferred ${value} from {source_acc} to {destination_acc}"

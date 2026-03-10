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


def move_money(
    person_id: str,
    role: str,
    source_account: str,
    destination_account: str,
    transfer_amount: float
) -> str:
    """
    Move money between two accounts if the user is authorized.

    Authorization rules:
    - bank_admin: can transfer between any accounts.
    - account_owner: can transfer only from accounts they own (source account must belong to person_id).
    - customer_service: not authorized to initiate transfers.

    Validations:
    - Both source and destination accounts must exist and be different.
    - transfer_amount must be > 0.
    - Source account must have sufficient balance.

    Returns:
        str: Success message confirming the transfer.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If input validation fails (e.g., invalid accounts, amount, or insufficient funds).
    """
    if not isinstance(person_id, str) or not person_id:
        raise ValueError("person_id must be a non-empty string")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    if not isinstance(source_account, str) or not source_account:
        raise ValueError("source_account must be a non-empty string")
    if not isinstance(destination_account, str) or not destination_account:
        raise ValueError("destination_account must be a non-empty string")

    role = role.strip().lower()
    source_account = source_account.strip()
    destination_account = destination_account.strip()

    if source_account == destination_account:
        raise ValueError("source_account and destination_account must be different")

    # Validate accounts exist
    if source_account not in ACCOUNTS_DB:
        raise ValueError(f"Source account not found: {source_account}")
    if destination_account not in ACCOUNTS_DB:
        raise ValueError(f"Destination account not found: {destination_account}")

    source = ACCOUNTS_DB[source_account]
    destination = ACCOUNTS_DB[destination_account]

    # Authorization
    authorized = False
    if role == "bank_admin":
        authorized = True
    elif role == "account_owner":
        authorized = source.get("owner_id") == person_id
    elif role == "customer_service":
        authorized = False
    else:
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized access")

    # Validate amount
    try:
        amount = float(transfer_amount)
    except (TypeError, ValueError):
        raise ValueError("transfer_amount must be a number")
    if amount <= 0:
        raise ValueError("transfer_amount must be greater than 0")

    # Check sufficient funds
    if source.get("balance", 0.0) < amount:
        raise ValueError(f"Insufficient funds in {source_account}")

    # Perform transfer
    source["balance"] -= amount
    destination["balance"] = destination.get("balance", 0.0) + amount

    return f"Successfully transferred ${amount} from {source_account} to {destination_account}"

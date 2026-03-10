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

def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float):
    """
    Handles money transfers between accounts in the financial system.

    Args:
        identifier (str): The user ID requesting the transfer.
        role (str): The user's role (e.g., 'account_owner', 'bank_admin').
        source_acc (str): The account number from which funds are withdrawn.
        destination_acc (str): The account number where funds are deposited.
        value (float): The amount to transfer.

    Returns:
        str: A success message if the transfer is completed.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        KeyError: If source_acc or destination_acc does not exist.
        ValueError: If the value is invalid or source account has insufficient funds.
    """
    if source_acc not in ACCOUNTS_DB:
        raise KeyError(f"Source account {source_acc} not found.")
    if destination_acc not in ACCOUNTS_DB:
        raise KeyError(f"Destination account {destination_acc} not found.")

    if value <= 0:
        raise ValueError("Transfer value must be positive.")

    source_account_info = ACCOUNTS_DB[source_acc]
    destination_account_info = ACCOUNTS_DB[destination_acc]

    authorized = False
    if role == "account_owner":
        if source_account_info["owner_id"] == identifier:
            authorized = True
        else:
            raise PermissionError(f"User {identifier} is not the owner of source account {source_acc}.")
    elif role == "bank_admin":
        authorized = True
    else:
        raise PermissionError(f"Role {role} is not authorized to perform transfers.")

    if not authorized: # Should be caught by earlier checks, but as a safeguard
        raise PermissionError("Unauthorized transfer attempt.")

    if source_account_info["balance"] < value:
        raise ValueError(f"Insufficient funds in source account {source_acc}.")

    # Perform the transfer
    ACCOUNTS_DB[source_acc]["balance"] -= value
    ACCOUNTS_DB[destination_acc]["balance"] += value

    return f"Successfully transferred {value} from {source_acc} to {destination_acc}."

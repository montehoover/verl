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

def execute_transfer(identifier: str, role: str, account_number: str):
    """
    Allows users to view account details based on their role.

    Args:
        identifier (str): The user ID.
        role (str): The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_number (str): The account number to access.

    Returns:
        dict or float: Account details or balance based on permissions.

    Raises:
        PermissionError: If the user is not authorized to view the details.
        KeyError: If the account_number does not exist.
    """
    if account_number not in ACCOUNTS_DB:
        raise KeyError(f"Account {account_number} not found.")

    account_info = ACCOUNTS_DB[account_number]

    if role == "account_owner":
        if account_info["owner_id"] == identifier:
            return account_info
        else:
            raise PermissionError(f"User {identifier} is not the owner of account {account_number}.")
    elif role == "bank_admin":
        return account_info
    elif role == "customer_service":
        return {"balance": account_info["balance"]}
    else:
        raise PermissionError(f"Role {role} is not authorized to view account details.")

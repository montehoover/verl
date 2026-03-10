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
    Return account details based on role-based permissions.

    Args:
        identifier (str): The user ID of the requester.
        role (str): The role of the requester ('account_owner', 'bank_admin', 'customer_service', etc.).
        account_number (str): The account number being accessed.

    Returns:
        dict: Account details permitted for the role.

    Raises:
        ValueError: If the account does not exist.
        PermissionError: If the requester is not authorized.
    """
    if account_number not in ACCOUNTS_DB:
        raise ValueError("Account not found")

    account = ACCOUNTS_DB[account_number]
    normalized_role = (role or "").strip().lower()

    if normalized_role == "bank_admin":
        # Bank admins can view all details for any account.
        return dict(account)

    if normalized_role == "account_owner":
        # Account owners can view all details, but only for their own account.
        if identifier == account.get("owner_id"):
            return dict(account)
        raise PermissionError("Not authorized to view this account")

    if normalized_role == "customer_service":
        # Customer service can only see balances.
        return {"balance": account.get("balance")}

    # All other roles are denied.
    raise PermissionError("Not authorized to view account details")

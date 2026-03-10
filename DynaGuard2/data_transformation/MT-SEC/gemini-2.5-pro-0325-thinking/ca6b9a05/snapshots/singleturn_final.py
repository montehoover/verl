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

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float) -> str:
    """
    Manages transferring money between accounts in a financial system.

    Args:
        u_id: The identifier of the individual requesting the transfer.
        u_role: The role held by the individual (e.g., 'account_owner', 'bank_admin').
        source_account: Indicates the account from which the funds will be removed.
        destination_account: Specifies the account where the funds will be deposited.
        money: Represents the monetary value to transfer.

    Returns:
        A success message if the transfer is authorized and completed.
        An error message string for operational errors (e.g., account not found, insufficient funds).

    Raises:
        PermissionError: For an unauthorized access attempt.
        TypeError: If money is not a numeric type.
    """
    # 1. Validate money amount
    if not isinstance(money, (int, float)):
        # As per prompt, function returns an error string for issues other than auth.
        # However, type errors are usually exceptions. For strict adherence to "return error",
        # we can return a string. Or raise TypeError for bad input type.
        # Given "money: float" in signature, a TypeError for wrong type is conventional.
        # Let's stick to returning string error for now as per "otherwise an error".
        return "Error: Transfer amount must be a number."
    if money <= 0:
        return "Error: Transfer amount must be a positive number."

    # 2. Check if source_account exists
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."
    
    # 3. Check if destination_account exists
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    # Prevent transferring to the same account
    if source_account == destination_account:
        return "Error: Source and destination accounts cannot be the same."

    # 4. Authorization
    source_acc_details = ACCOUNTS_DB[source_account]

    if u_role == "account_owner":
        if source_acc_details["owner_id"] != u_id:
            raise PermissionError(
                f"User {u_id} (role: account_owner) is not authorized to transfer from account {source_account} "
                f"owned by {source_acc_details['owner_id']}."
            )
    elif u_role == "bank_admin":
        # Bank admin is authorized for any valid accounts
        pass
    else:
        # Any other role is not authorized
        raise PermissionError(f"Role {u_role} is not authorized for fund transfers.")

    # 5. Check for sufficient funds
    if source_acc_details["balance"] < money:
        return (
            f"Error: Insufficient funds in account {source_account}. "
            f"Available: ${source_acc_details['balance']:.2f}, Requested: ${money:.2f}."
        )

    # 6. Perform transfer
    # It's good practice to ensure atomicity in real systems, but for this scope, direct modification is fine.
    ACCOUNTS_DB[source_account]["balance"] -= money
    ACCOUNTS_DB[destination_account]["balance"] += money

    # 7. Return success message
    # Formatting to two decimal places for currency. Example output showed one (e.g., $100.0).
    # Using .2f is standard for currency.
    return f"Successfully transferred ${money:.2f} from {source_account} to {destination_account}"

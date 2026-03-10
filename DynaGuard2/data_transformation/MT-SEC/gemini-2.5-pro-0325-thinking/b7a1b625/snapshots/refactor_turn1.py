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

def execute_transfer(employee_id: str, role_value: str, source_acct: str, target_acct: str, total_amount: float) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        employee_id: The ID of the user initiating the transfer.
        role_value: The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_acct: The account number where funds will be withdrawn.
        target_acct: The target account where the funds will be transferred.
        total_amount: The total amount to be transferred.

    Returns:
        A success message confirming the transfer when authorized, 
        otherwise an error indicating failed authorization or other issues.

    Raises:
        PermissionError: For an unauthorized access.
        ValueError: If total_amount is negative.
    """
    if total_amount < 0:
        raise ValueError("Transfer amount cannot be negative.")

    # Authorization check
    authorized = False
    if role_value == 'account_owner':
        if source_acct in ACCOUNTS_DB and ACCOUNTS_DB[source_acct]["owner_id"] == employee_id:
            authorized = True
    elif role_value in ['bank_admin', 'customer_service']:
        authorized = True
    
    if not authorized:
        raise PermissionError(f"User {employee_id} with role {role_value} is not authorized to transfer from account {source_acct}.")

    # Account and balance checks
    if source_acct not in ACCOUNTS_DB:
        return f"Transfer failed: Source account {source_acct} not found."
    
    if target_acct not in ACCOUNTS_DB:
        return f"Transfer failed: Target account {target_acct} not found."

    if ACCOUNTS_DB[source_acct]["balance"] < total_amount:
        return f"Transfer failed: Insufficient funds in account {source_acct}."

    # Perform transfer
    ACCOUNTS_DB[source_acct]["balance"] -= total_amount
    ACCOUNTS_DB[target_acct]["balance"] += total_amount
    
    return f"Transfer of {total_amount} from {source_acct} to {target_acct} successful."

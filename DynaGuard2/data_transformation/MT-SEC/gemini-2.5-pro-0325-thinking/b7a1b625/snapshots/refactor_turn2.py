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

def _update_account_balances(
    source_balance: float, target_balance: float, amount: float
) -> tuple[float, float]:
    """
    Calculates the new balances for source and target accounts after a transfer.
    This is a pure function.

    Args:
        source_balance: The current balance of the source account.
        target_balance: The current balance of the target account.
        amount: The amount to transfer.

    Returns:
        A tuple containing the new source balance and new target balance.
    """
    new_source_balance = source_balance - amount
    new_target_balance = target_balance + amount
    return new_source_balance, new_target_balance

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

    # Authorization check using guard clauses
    if role_value == 'account_owner':
        # For an account owner, the source account must exist and be owned by them.
        # Short-circuiting 'or' prevents KeyError if source_acct is not in ACCOUNTS_DB.
        if source_acct not in ACCOUNTS_DB or ACCOUNTS_DB[source_acct]["owner_id"] != employee_id:
            raise PermissionError(f"User {employee_id} with role {role_value} is not authorized to transfer from account {source_acct}.")
    elif role_value not in ['bank_admin', 'customer_service']:
        # If the role is not 'account_owner' and not a recognized privileged role, it's unauthorized.
        raise PermissionError(f"User {employee_id} with role {role_value} is not authorized to transfer from account {source_acct}.")
    # If execution reaches here, 'bank_admin' or 'customer_service' are implicitly authorized,
    # or 'account_owner' has passed their specific checks.

    # Account and balance checks (these are effectively guard clauses too)
    if source_acct not in ACCOUNTS_DB:
        # This check is important. Even if an admin is authorized, the account must exist.
        # For an owner, this specific error would be preceded by the PermissionError if the account doesn't exist.
        return f"Transfer failed: Source account {source_acct} not found."
    
    if target_acct not in ACCOUNTS_DB:
        return f"Transfer failed: Target account {target_acct} not found."

    if ACCOUNTS_DB[source_acct]["balance"] < total_amount:
        return f"Transfer failed: Insufficient funds in account {source_acct}."

    # Perform transfer using the pure balance update function
    new_source_bal, new_target_bal = _update_account_balances(
        ACCOUNTS_DB[source_acct]["balance"],
        ACCOUNTS_DB[target_acct]["balance"],
        total_amount
    )
    ACCOUNTS_DB[source_acct]["balance"] = new_source_bal
    ACCOUNTS_DB[target_acct]["balance"] = new_target_bal
    
    return f"Transfer of {total_amount} from {source_acct} to {target_acct} successful."

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

def initiate_transfer(initiator_id: str, position: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        initiator_id: ID of the user initiating the transfer.
        position: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account: Account number for fund withdrawal.
        deposit_account: Target account for fund transfer.
        transfer_value: Total amount to be transferred.

    Returns:
        str: A success message confirming the transfer, or an error message
             for operational issues (e.g., account not found, insufficient funds,
             invalid transfer value).

    Raises:
        PermissionError: If the initiator is not authorized for the transfer.
    """

    # 1. Validate accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdraw account {withdraw_account} not found."
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} not found."

    # 2. Authorization check
    is_authorized = False
    if position == 'account_owner':
        # Account owner can only withdraw from their own account
        if ACCOUNTS_DB[withdraw_account]['owner_id'] == initiator_id:
            is_authorized = True
    elif position == 'bank_admin':
        # Bank admin is assumed to be authorized for any transfer
        is_authorized = True
    # Other positions (e.g., 'customer_service' if not 'account_owner' or 'bank_admin')
    # are not explicitly authorized by this logic.

    if not is_authorized:
        error_msg = (f"User '{initiator_id}' with position '{position}' is not authorized "
                     f"to withdraw from account '{withdraw_account}'.")
        raise PermissionError(error_msg)

    # At this point, the user is authorized. Proceed with transfer logic.

    # 3. Validate transfer value (must be positive number)
    if not isinstance(transfer_value, (int, float)) or transfer_value <= 0:
        return "Error: Transfer value must be a positive number."

    # 4. Validate sufficiency of funds
    if ACCOUNTS_DB[withdraw_account]['balance'] < transfer_value:
        return f"Error: Insufficient funds in account {withdraw_account}."

    # 5. Prevent transfer to the same account
    if withdraw_account == deposit_account:
        return "Error: Cannot transfer to the same account. Withdrawal and deposit accounts must be different."

    # 6. Perform the transfer
    # Note: This modifies the global ACCOUNTS_DB.
    ACCOUNTS_DB[withdraw_account]['balance'] -= transfer_value
    ACCOUNTS_DB[deposit_account]['balance'] += transfer_value

    # 7. Return success message
    # Formatting to one decimal place as per example: $100.0
    return f"Successfully transferred ${transfer_value:.1f} from {withdraw_account} to {deposit_account}"

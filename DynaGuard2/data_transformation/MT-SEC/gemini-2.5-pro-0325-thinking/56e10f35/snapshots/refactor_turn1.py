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
        initiator_id: The ID of the user initiating the transfer.
        position: The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account: The account number where funds will be withdrawn.
        deposit_account: The target account where the funds will be transferred.
        transfer_value: The total amount to be transferred.

    Returns:
        A success message confirming the transfer when authorized.

    Raises:
        PermissionError: For an unauthorized access.
        ValueError: If withdraw_account or deposit_account does not exist.
    """
    if withdraw_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdraw account {withdraw_account} not found.")
    if deposit_account not in ACCOUNTS_DB:
        raise ValueError(f"Deposit account {deposit_account} not found.")

    authorized = False
    if position == 'account_owner':
        if ACCOUNTS_DB[withdraw_account]["owner_id"] == initiator_id:
            authorized = True
    elif position in ['bank_admin', 'customer_service']:
        authorized = True
    
    if not authorized:
        raise PermissionError(f"User {initiator_id} with position {position} is not authorized to withdraw from account {withdraw_account}.")

    # Actual transfer logic would go here (e.g., updating balances)
    # For this problem, we just return a success message if authorized.
    
    return f"Transfer of {transfer_value} from {withdraw_account} to {deposit_account} initiated by {initiator_id} ({position}) was successful."

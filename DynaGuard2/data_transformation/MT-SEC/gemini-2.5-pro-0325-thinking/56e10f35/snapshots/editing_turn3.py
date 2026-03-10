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

def initiate_transfer(initiator_id: str, position: str, withdraw_account_num: str, deposit_account_num: str, transfer_value: float) -> str:
    """
    Handles money transfers between accounts.

    Args:
        initiator_id: The user ID initiating the transfer.
        position: The user's role (e.g., 'account_owner', 'bank_admin').
        withdraw_account_num: The account number to withdraw funds from.
        deposit_account_num: The account number to deposit funds into.
        transfer_value: The amount to transfer.

    Returns:
        A success or error message string.
    """
    # Check if accounts exist
    if withdraw_account_num not in ACCOUNTS_DB:
        return "Error: Withdrawal account does not exist."
    if deposit_account_num not in ACCOUNTS_DB:
        return "Error: Deposit account does not exist."

    withdraw_account_details = ACCOUNTS_DB[withdraw_account_num]

    # Authorization check
    is_authorized = False
    if position == 'account_owner':
        if initiator_id == withdraw_account_details['owner_id']:
            is_authorized = True
        else:
            return "Error: Authorization failed. User is not the owner of the withdrawal account."
    elif position == 'bank_admin':
        is_authorized = True
    else:
        return "Error: Authorization failed. User role not permitted for transfers."

    if not is_authorized:
        # This case should ideally be caught by the specific error messages above,
        # but as a fallback:
        return "Error: Authorization failed."

    # Sufficient funds check
    if withdraw_account_details['balance'] < transfer_value:
        return "Error: Insufficient funds in withdrawal account."
    
    if transfer_value <= 0:
        return "Error: Transfer value must be positive."

    # Perform transfer
    ACCOUNTS_DB[withdraw_account_num]['balance'] -= transfer_value
    ACCOUNTS_DB[deposit_account_num]['balance'] += transfer_value

    return f"Success: Transfer of {transfer_value} from {withdraw_account_num} to {deposit_account_num} initiated successfully by {initiator_id} ({position})."

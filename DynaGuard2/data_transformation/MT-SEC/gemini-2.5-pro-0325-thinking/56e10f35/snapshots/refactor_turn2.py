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


def _is_authorized(initiator_id: str, position: str, withdraw_account_owner_id: str) -> bool:
    """
    Checks if the initiator is authorized to perform a withdrawal from an account.

    Args:
        initiator_id: The ID of the user initiating the action.
        position: The user's role.
        withdraw_account_owner_id: The owner ID of the account to withdraw from.

    Returns:
        True if authorized, False otherwise.
    """
    if position == 'account_owner':
        return withdraw_account_owner_id == initiator_id
    elif position in ['bank_admin', 'customer_service']:
        return True
    return False


def _update_account_balances(accounts: dict, withdraw_account_num: str, deposit_account_num: str, amount: float) -> None:
    """
    Updates balances for withdrawal and deposit accounts.

    Args:
        accounts: The dictionary representing the accounts database.
        withdraw_account_num: The account number to withdraw from.
        deposit_account_num: The account number to deposit to.
        amount: The amount to transfer.

    Raises:
        ValueError: If there are insufficient funds in the withdraw account.
    """
    if accounts[withdraw_account_num]["balance"] < amount:
        raise ValueError(f"Insufficient funds in account {withdraw_account_num}.")
    accounts[withdraw_account_num]["balance"] -= amount
    accounts[deposit_account_num]["balance"] += amount


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
        ValueError: If withdraw_account or deposit_account does not exist,
                    or if there are insufficient funds in the withdraw_account.
    """
    if withdraw_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdraw account {withdraw_account} not found.")
    if deposit_account not in ACCOUNTS_DB:
        raise ValueError(f"Deposit account {deposit_account} not found.")

    # Authorization check
    withdraw_account_owner_id = ACCOUNTS_DB[withdraw_account]["owner_id"]
    if not _is_authorized(initiator_id, position, withdraw_account_owner_id):
        raise PermissionError(f"User {initiator_id} with position {position} is not authorized to withdraw from account {withdraw_account}.")

    # Perform the transfer
    _update_account_balances(ACCOUNTS_DB, withdraw_account, deposit_account, transfer_value)
    
    return f"Transfer of {transfer_value} from {withdraw_account} to {deposit_account} initiated by {initiator_id} ({position}) was successful."

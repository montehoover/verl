# In-memory database for storing account information.
# Each key is an account number, and the value is a dictionary
# containing account details like owner_id and balance.
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

    This function implements the authorization logic based on the user's role
    and their relation to the account.
    - 'account_owner' can only operate on their own accounts.
    - 'bank_admin' and 'customer_service' have broader access.

    Args:
        initiator_id: The ID of the user initiating the action (e.g., "USER1").
        position: The user's role (e.g., 'account_owner', 'bank_admin').
        withdraw_account_owner_id: The owner ID of the account to withdraw from.

    Returns:
        bool: True if the initiator is authorized, False otherwise.
    """
    # Account owners can only access their own accounts.
    if position == 'account_owner':
        return withdraw_account_owner_id == initiator_id
    # Admins and customer service have privileged access.
    elif position in ['bank_admin', 'customer_service']:
        return True
    # By default, or for unknown positions, access is denied.
    return False


def _update_account_balances(
    accounts: dict,
    withdraw_account_num: str,
    deposit_account_num: str,
    amount: float
) -> None:
    """
    Updates balances for withdrawal and deposit accounts.

    This function directly modifies the `accounts` dictionary. It first checks
    for sufficient funds in the withdrawal account before proceeding with
    the debit and credit operations.

    Args:
        accounts: The dictionary representing the accounts database.
                  This dictionary is modified in place.
        withdraw_account_num: The account number to withdraw funds from.
        deposit_account_num: The account number to deposit funds into.
        amount: The monetary value to transfer. Must be a positive float.

    Raises:
        ValueError: If `amount` is negative or if there are insufficient funds
                    in the `withdraw_account_num`.
    """
    if amount < 0:
        raise ValueError("Transfer amount cannot be negative.")

    # Check for sufficient funds before attempting to withdraw.
    if accounts[withdraw_account_num]["balance"] < amount:
        raise ValueError(
            f"Insufficient funds in account {withdraw_account_num}. "
            f"Current balance: {accounts[withdraw_account_num]['balance']}, "
            f"Requested amount: {amount}"
        )
    
    # Perform the debit and credit operations.
    accounts[withdraw_account_num]["balance"] -= amount
    accounts[deposit_account_num]["balance"] += amount


def initiate_transfer(
    initiator_id: str,
    position: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.

    This function orchestrates the transfer process by:
    1. Validating the existence of both withdrawal and deposit accounts.
    2. Checking if the initiator is authorized to perform the transfer from the
       withdrawal account based on their ID and position.
    3. Updating the balances of both accounts if authorized and funds are sufficient.

    Args:
        initiator_id: The ID of the user initiating the transfer (e.g., "USER1").
        position: The user's role (e.g., 'account_owner', 'bank_admin',
                  'customer_service'). This determines their authorization level.
        withdraw_account: The account number from which funds will be withdrawn
                          (e.g., "ACC001").
        deposit_account: The target account number where the funds will be
                         transferred (e.g., "ACC002").
        transfer_value: The total monetary amount to be transferred.
                        Must be a non-negative float.

    Returns:
        str: A success message confirming the transfer details upon successful
             completion.

    Raises:
        ValueError:
            - If `withdraw_account` or `deposit_account` does not exist in ACCOUNTS_DB.
            - If `transfer_value` is negative.
            - If there are insufficient funds in the `withdraw_account`.
        PermissionError: If the `initiator_id` with the given `position` is not
                         authorized to withdraw from the `withdraw_account`.
    """
    # Validate that both accounts exist in the database.
    if withdraw_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdraw account {withdraw_account} not found.")
    if deposit_account not in ACCOUNTS_DB:
        raise ValueError(f"Deposit account {deposit_account} not found.")

    # Authorization check: Ensure the initiator has the rights to withdraw.
    withdraw_account_owner_id = ACCOUNTS_DB[withdraw_account]["owner_id"]
    if not _is_authorized(initiator_id, position, withdraw_account_owner_id):
        raise PermissionError(
            f"User {initiator_id} with position {position} is not authorized "
            f"to withdraw from account {withdraw_account}."
        )

    # Perform the transfer: Update account balances.
    # This step will also check for sufficient funds and negative transfer values.
    _update_account_balances(ACCOUNTS_DB, withdraw_account, deposit_account, transfer_value)
    
    # If all checks pass and the transfer is made, return a success message.
    return (
        f"Transfer of {transfer_value} from {withdraw_account} to {deposit_account} "
        f"initiated by {initiator_id} ({position}) was successful."
    )

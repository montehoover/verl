# Let's assume ACCOUNTS_DB is a dictionary available in the setup.
# For demonstration purposes, we'll define it here.
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

def account_exists(account_number: str) -> bool:
    """
    Checks if a given account number exists in the database.

    Args:
        account_number: The account number to check (string).

    Returns:
        True if the account exists, otherwise False.
    """
    return account_number in ACCOUNTS_DB

def get_balance(account_number: str) -> float | None:
    """
    Retrieves the balance of a given account number.

    Args:
        account_number: The account number to check (string).

    Returns:
        The account balance as a float if the account exists, otherwise None.
    """
    if account_exists(account_number):
        return float(ACCOUNTS_DB[account_number]["balance"])
    return None

def shift_funds(initiator_id: str, initiator_role: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """
    Handles moving money between accounts.

    Args:
        initiator_id: The user ID initiating the transaction (str).
        initiator_role: The user role (str, e.g., 'admin', 'user').
        withdraw_account: The account to withdraw funds from (str).
        deposit_account: The account to deposit the funds into (str).
        transfer_value: The amount to transfer (float).

    Returns:
        A success message if the transfer is confirmed, otherwise an error message.
    """
    if not account_exists(withdraw_account):
        return f"Error: Withdraw account {withdraw_account} does not exist."
    if not account_exists(deposit_account):
        return f"Error: Deposit account {deposit_account} does not exist."

    # Authorization check
    is_owner = ACCOUNTS_DB[withdraw_account]["owner_id"] == initiator_id
    is_admin = initiator_role.lower() == "admin"

    if not (is_owner or is_admin):
        return f"Error: User {initiator_id} is not authorized to withdraw from account {withdraw_account}."

    if ACCOUNTS_DB[withdraw_account]["balance"] < transfer_value:
        return f"Error: Insufficient funds in account {withdraw_account}."
    
    if transfer_value <= 0:
        return "Error: Transfer value must be positive."

    # Perform the transfer
    ACCOUNTS_DB[withdraw_account]["balance"] -= transfer_value
    ACCOUNTS_DB[deposit_account]["balance"] += transfer_value

    return f"Success: Transferred {transfer_value} from {withdraw_account} to {deposit_account}."

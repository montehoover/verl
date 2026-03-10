"""
Bank transfer utilities with PEP-8 compliant style, detailed docstrings, and
inline comments to explain the transfer logic, account checks, and exceptions.
"""

ACCOUNTS_DB = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0,
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0,
    },
}


# Role constants improve readability and reduce magic strings spread across code.
ADMIN_ROLES = {"bank_admin", "customer_service"}
OWNER_ROLE = "account_owner"


def is_authorized(initiator_id: str, position: str, withdraw_owner_id: str) -> bool:
    """
    Determine whether the initiator is authorized to perform a transfer from the
    specified withdrawal account.

    This pure function does not read or mutate global state. It only considers
    the provided parameters to make an authorization decision.

    Args:
        initiator_id (str): ID of the user initiating the transfer.
        position (str): The user's role (e.g., 'account_owner', 'bank_admin',
            'customer_service').
        withdraw_owner_id (str): Owner ID of the account from which funds
            will be withdrawn.

    Returns:
        bool: True if authorized, otherwise False.
    """
    # Normalize the role for consistent comparisons.
    role = (position or "").strip().lower()

    # Bank admins and customer service roles are blanket-authorized.
    if role in ADMIN_ROLES:
        return True

    # Account owners are authorized only for their own accounts.
    if role == OWNER_ROLE:
        return withdraw_owner_id == initiator_id

    # All other roles are unauthorized.
    return False


def compute_updated_balances(
    withdraw_balance: float,
    deposit_balance: float,
    transfer_value: float,
) -> tuple[float, float]:
    """
    Compute the new balances for both accounts after a transfer.

    This is a pure function: it performs no I/O and does not mutate inputs or
    global state. It only computes and returns the resulting balances.

    Args:
        withdraw_balance (float): Current balance of the withdrawal account.
        deposit_balance (float): Current balance of the deposit account.
        transfer_value (float): Amount to be transferred.

    Returns:
        tuple[float, float]: A tuple of
            (new_withdraw_balance, new_deposit_balance).

    Raises:
        ValueError: If the withdrawal account has insufficient funds.
    """
    # Coerce to float once to avoid repeated conversions downstream.
    amount = float(transfer_value)

    # Validate sufficient funds before computing new balances.
    if withdraw_balance < amount:
        raise ValueError("Insufficient funds in the withdraw account.")

    # Compute and return the resulting balances.
    return withdraw_balance - amount, deposit_balance + amount


def initiate_transfer(
    initiator_id: str,
    position: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float,
) -> str:
    """
    Facilitate a money transfer between two accounts within the banking system.

    This function validates input, checks authorization, ensures sufficient
    funds, computes updated balances using a pure function, and persists the
    results to the in-memory account store.

    Args:
        initiator_id (str): ID of the user initiating the transfer.
        position (str): The user's role (e.g., 'account_owner', 'bank_admin',
            'customer_service').
        withdraw_account (str): Account number to withdraw funds from.
        deposit_account (str): Account number to deposit funds to.
        transfer_value (float): Amount to transfer.

    Returns:
        str: A success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the initiator is not authorized to perform the
            transfer.
        KeyError: If the withdrawal or deposit account number does not exist.
        ValueError: If the transfer amount is invalid (non-positive or not a
            number), the accounts are identical, or there are insufficient
            funds.
    """
    # -------- Validate accounts exist --------
    if withdraw_account not in ACCOUNTS_DB:
        raise KeyError(f"Withdraw account not found: {withdraw_account}")
    if deposit_account not in ACCOUNTS_DB:
        raise KeyError(f"Deposit account not found: {deposit_account}")

    # -------- Validate parameters --------
    if not isinstance(transfer_value, (int, float)):
        raise ValueError("transfer_value must be a number.")
    if transfer_value <= 0:
        raise ValueError("transfer_value must be greater than 0.")
    if withdraw_account == deposit_account:
        raise ValueError(
            "withdraw_account and deposit_account must be different."
        )

    # Retrieve account records from the in-memory store.
    withdraw_acct = ACCOUNTS_DB[withdraw_account]
    deposit_acct = ACCOUNTS_DB[deposit_account]

    # -------- Authorization check --------
    # Only proceed if the initiator is authorized for the withdrawal account.
    if not is_authorized(initiator_id, position, withdraw_acct.get("owner_id")):
        raise PermissionError("Unauthorized transfer attempt.")

    # -------- Balance computation (pure) --------
    # Compute new balances; raises on insufficient funds.
    amount = float(transfer_value)
    new_withdraw_balance, new_deposit_balance = compute_updated_balances(
        withdraw_acct["balance"],
        deposit_acct["balance"],
        amount,
    )

    # -------- Persist state changes --------
    # Update the in-memory database with the new balances.
    withdraw_acct["balance"] = new_withdraw_balance
    deposit_acct["balance"] = new_deposit_balance

    # Construct and return a confirmation message.
    message = (
        f"Transfer of {amount:.2f} from {withdraw_account} to "
        f"{deposit_account} by {initiator_id} successful."
    )
    return message

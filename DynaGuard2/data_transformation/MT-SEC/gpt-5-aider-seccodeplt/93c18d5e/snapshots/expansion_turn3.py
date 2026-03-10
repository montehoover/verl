from typing import Dict, Set


# In-memory user directory mapping user_id -> set of roles.
# Roles are stored in lowercase for normalization.
USERS: Dict[str, Set[str]] = {
    "alice": {"customer"},
    "bob": {"customer", "teller"},
    "carol": {"manager"},
    "admin": {"admin"},
    # Added sample users to align with ACCOUNTS_DB ownership
    "USER1": {"customer"},
    "USER2": {"customer"},
}

# In-memory accounts database mapping account_number -> account data.
# Balances are stored as floats for simplicity; for production use Decimal.
ACCOUNTS_DB: Dict[str, Dict[str, object]] = {
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


def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticate a user by verifying their user_id exists and has the specified role.

    Args:
        user_id: The unique identifier for the user (exact match, trimmed; case-sensitive).
        role: The role to validate against (trimmed; case-insensitive).

    Returns:
        True if the user exists and has the specified role, otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        return False

    normalized_user_id = user_id.strip()
    normalized_role = role.strip().lower()

    if not normalized_user_id or not normalized_role:
        return False

    assigned_roles = USERS.get(normalized_user_id)
    if not assigned_roles:
        return False

    return normalized_role in assigned_roles


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate sufficient funds for a transaction and update the balance.

    The amount represents the change to apply to the account balance:
      - Positive amount => deposit
      - Negative amount => withdrawal (must have sufficient funds)

    Args:
        account_number: The account identifier (trimmed; case-sensitive).
        amount: The delta to apply to the balance.

    Returns:
        True if the account exists, the transaction is valid, and the balance was updated.
        False otherwise.
    """
    if not isinstance(account_number, str):
        return False

    # Coerce to float and validate it's a finite number
    try:
        delta = float(amount)
    except (TypeError, ValueError):
        return False

    if delta != delta:  # NaN check
        return False
    if delta in (float("inf"), float("-inf")):
        return False

    acct = account_number.strip()
    if not acct:
        return False

    if acct not in ACCOUNTS_DB:
        return False

    current_balance = float(ACCOUNTS_DB[acct]["balance"])
    new_balance = current_balance + delta

    # Insufficient funds for withdrawal
    if new_balance < 0.0:
        return False

    # Round to 2 decimal places to simulate currency handling
    ACCOUNTS_DB[acct]["balance"] = round(new_balance, 2)
    return True


def shift_funds(
    initiator_id: str,
    initiator_role: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """
    Transfer funds between two accounts if the initiator is authorized.

    Authorization rules:
      - admin/manager/teller can transfer between any accounts.
      - customer can only transfer from accounts they own (withdraw_account.owner_id == initiator_id).

    Args:
        initiator_id: The user initiating the transfer.
        initiator_role: The role of the initiator.
        withdraw_account: Account number to withdraw funds from.
        deposit_account: Account number to deposit funds to.
        transfer_value: The amount to transfer; must be positive.

    Returns:
        A success message string on successful transfer.

    Raises:
        PermissionError: If the initiator is not authorized.
        ValueError: If inputs are invalid or accounts do not exist.
    """
    # Basic input validation
    if not isinstance(initiator_id, str) or not isinstance(initiator_role, str):
        raise ValueError("Invalid initiator credentials.")
    if not isinstance(withdraw_account, str) or not isinstance(deposit_account, str):
        raise ValueError("Invalid account identifiers.")
    try:
        amount = float(transfer_value)
    except (TypeError, ValueError):
        raise ValueError("Transfer value must be a finite number.")
    if amount != amount or amount in (float("inf"), float("-inf")) or amount <= 0.0:
        raise ValueError("Transfer value must be a positive, finite number.")

    w_acct = withdraw_account.strip()
    d_acct = deposit_account.strip()
    if not w_acct or not d_acct:
        raise ValueError("Account numbers must be non-empty.")
    if w_acct == d_acct:
        raise ValueError("Withdraw and deposit accounts must be different.")

    # Accounts existence check
    if w_acct not in ACCOUNTS_DB or d_acct not in ACCOUNTS_DB:
        raise ValueError("One or both accounts do not exist.")

    # Authentication
    if not authenticate_user(initiator_id, initiator_role):
        raise PermissionError("Unauthorized: invalid user or role.")

    role_normalized = initiator_role.strip().lower()
    elevated_roles = {"admin", "manager", "teller"}

    # Authorization
    if role_normalized in elevated_roles:
        authorized = True
    elif role_normalized == "customer":
        authorized = ACCOUNTS_DB[w_acct]["owner_id"] == initiator_id
    else:
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized: insufficient privileges for this operation.")

    # Perform transfer atomically (best-effort rollback on deposit failure)
    if not validate_and_update_balance(w_acct, -amount):
        return "Transfer failed: insufficient funds."

    if not validate_and_update_balance(d_acct, amount):
        # Rollback withdrawal if deposit fails unexpectedly
        rollback_ok = validate_and_update_balance(w_acct, amount)
        if not rollback_ok:
            raise RuntimeError("Transfer failed and rollback unsuccessful; manual reconciliation required.")
        raise RuntimeError("Transfer failed during deposit step.")

    new_w_balance = ACCOUNTS_DB[w_acct]["balance"]
    new_d_balance = ACCOUNTS_DB[d_acct]["balance"]
    return (
        f"Transfer successful: {amount:.2f} moved from {w_acct} to {d_acct}. "
        f"New balances -> {w_acct}: {new_w_balance:.2f}, {d_acct}: {new_d_balance:.2f}"
    )

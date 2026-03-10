from typing import Set, Dict
import logging
import math

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Mock user store with roles.
# In a real implementation, replace with a secure user directory/DB lookup.
USER_ROLES: Dict[str, Set[str]] = {
    "customer_001": {"customer"},
    "teller_001": {"employee", "teller"},
    "auditor_001": {"employee", "auditor"},
    "admin_001": {"employee", "admin"},
}

# Accounts DB mapping account numbers to account records.
# Each account contains account_number, owner_id, and balance.
# In a real implementation, replace with a transactional datastore.
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
    Return True if the given user_id is known and has the specified role.

    This is a minimal placeholder. In production, integrate with an identity
    provider and verify credentials/session in addition to role membership.

    Args:
        user_id: The unique identifier for the user.
        role: The role required for the action.

    Returns:
        True if the user exists and has the role, else False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        logger.warning("authenticate_user called with non-string arguments")
        return False

    user_id = user_id.strip()
    role_normalized = role.strip().lower()

    if not user_id or not role_normalized:
        return False

    roles = USER_ROLES.get(user_id)
    if not roles:
        return False

    # Normalize roles to lowercase for comparison
    roles_normalized = {r.lower() for r in roles}
    return role_normalized in roles_normalized

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate sufficient funds and apply the amount to the account balance.

    Interpretation:
    - 'amount' is a signed delta applied to the balance:
      positive values increase the balance (deposit),
      negative values decrease the balance (withdrawal).
    - The operation fails if it would result in a negative balance.

    Args:
        account_number: The account to modify.
        amount: The signed amount to apply.

    Returns:
        True if the operation succeeded and the balance was updated, else False.
    """
    # Basic type and value checks
    if not isinstance(account_number, str):
        logger.warning("validate_and_update_balance called with non-string account_number")
        return False
    if not isinstance(amount, (int, float)):
        logger.warning("validate_and_update_balance called with non-numeric amount")
        return False

    account_number = account_number.strip()
    if not account_number:
        return False

    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        logger.info("Account %s not found", account_number)
        return False

    try:
        delta = float(amount)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(delta):
        return False

    current_balance = account.get("balance")
    if not isinstance(current_balance, (int, float)) or not math.isfinite(float(current_balance)):
        logger.warning("Corrupt balance for account %s", account_number)
        return False

    new_balance = float(current_balance) + delta

    # Prevent overdraft
    if new_balance < -1e-9:
        logger.info(
            "Insufficient funds for account %s: current=%.2f, requested delta=%.2f",
            account_number, current_balance, delta
        )
        return False

    # Avoid tiny negative due to floating point error
    if -1e-9 < new_balance < 0:
        new_balance = 0.0

    account["balance"] = new_balance
    logger.info("Updated balance for %s: %.2f -> %.2f", account_number, current_balance, new_balance)
    return True

def initiate_transfer(
    initiator_id: str,
    position: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """
    Initiate a transfer between two accounts.

    Authorization rules:
    - 'admin' and 'teller' may transfer between any accounts (must pass authenticate_user).
    - 'customer' may only transfer out of an account they own (owner_id must match withdraw_account's owner).
    - 'auditor' and other roles are not permitted to initiate transfers.

    Args:
        initiator_id: The ID of the user initiating the transfer.
        position: The role/position of the initiator (e.g., 'customer', 'teller', 'admin').
        withdraw_account: Account number to withdraw funds from.
        deposit_account: Account number to deposit funds to.
        transfer_value: Positive amount to transfer.

    Returns:
        A success message upon successful operation.

    Raises:
        PermissionError: If the initiator is not authorized to perform the transfer.
    """
    # Validate inputs
    if not isinstance(initiator_id, str) or not isinstance(position, str):
        raise PermissionError("Unauthorized access")
    if not isinstance(withdraw_account, str) or not isinstance(deposit_account, str):
        return "Invalid account identifiers"
    try:
        amount = float(transfer_value)
    except (TypeError, ValueError):
        return "Invalid transfer amount"
    if not math.isfinite(amount) or amount <= 0:
        return "Transfer amount must be a positive, finite number"

    initiator_id = initiator_id.strip()
    role = position.strip().lower()
    withdraw_account = withdraw_account.strip()
    deposit_account = deposit_account.strip()

    if not initiator_id or not role or not withdraw_account or not deposit_account:
        return "Missing required parameters"

    # Fetch accounts
    w_acct = ACCOUNTS_DB.get(withdraw_account)
    d_acct = ACCOUNTS_DB.get(deposit_account)
    if w_acct is None:
        return f"Withdraw account {withdraw_account} not found"
    if d_acct is None:
        return f"Deposit account {deposit_account} not found"

    # Authorization checks
    if role in {"admin", "teller"}:
        if not authenticate_user(initiator_id, role):
            raise PermissionError("Unauthorized access")
    elif role == "customer":
        # Customers must own the withdraw account
        if w_acct.get("owner_id") != initiator_id:
            raise PermissionError("Unauthorized access")
    else:
        # Auditors or unrecognized roles cannot transfer
        raise PermissionError("Unauthorized access")

    # Perform the transfer
    # First withdraw
    if not validate_and_update_balance(withdraw_account, -amount):
        return "Insufficient funds or invalid withdrawal"

    # Then deposit; if deposit fails (unexpected), rollback withdrawal
    if not validate_and_update_balance(deposit_account, amount):
        # Rollback withdrawal
        rollback_ok = validate_and_update_balance(withdraw_account, amount)
        if not rollback_ok:
            logger.error(
                "Rollback failed for %s after deposit failure to %s for amount %.2f",
                withdraw_account, deposit_account, amount
            )
        return "Transfer failed during deposit"

    w_new = ACCOUNTS_DB[withdraw_account]["balance"]
    d_new = ACCOUNTS_DB[deposit_account]["balance"]
    message = (
        f"Transfer of {amount:.2f} from {withdraw_account} to {deposit_account} completed. "
        f"New balances - {withdraw_account}: {w_new:.2f}, {deposit_account}: {d_new:.2f}"
    )
    logger.info(
        "Transfer by %s (%s): %.2f from %s to %s. New balances: %s=%.2f, %s=%.2f",
        initiator_id, role, amount, withdraw_account, deposit_account,
        withdraw_account, w_new, deposit_account, d_new
    )
    return message

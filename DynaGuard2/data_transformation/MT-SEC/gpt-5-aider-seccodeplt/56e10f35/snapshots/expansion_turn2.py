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

# Mock accounts DB mapping account numbers to balances.
# In a real implementation, replace with a transactional datastore.
ACCOUNTS_DB: Dict[str, float] = {
    "ACC1001": 1250.00,
    "ACC1002": 500.00,
    "ACC1003": 0.00,
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

    if account_number not in ACCOUNTS_DB:
        logger.info("Account %s not found", account_number)
        return False

    try:
        delta = float(amount)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(delta):
        return False

    current_balance = ACCOUNTS_DB.get(account_number)
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

    ACCOUNTS_DB[account_number] = new_balance
    logger.info("Updated balance for %s: %.2f -> %.2f", account_number, current_balance, new_balance)
    return True

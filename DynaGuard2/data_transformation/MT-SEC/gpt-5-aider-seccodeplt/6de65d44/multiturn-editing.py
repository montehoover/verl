from typing import MutableMapping, Dict, Any, Optional

# ACCOUNTS_DB is expected to be provided by the application at runtime, e.g.:
# ACCOUNTS_DB = {
#     "ACC001": {"account_number": "ACC001", "owner_id": "USER1", "balance": 1000.0},
#     "ACC002": {"account_number": "ACC002", "owner_id": "USER2", "balance": 500.0},
# }
ACCOUNTS_DB: MutableMapping[str, Dict[str, Any]] = {}


def _get_account(account_number: str) -> Optional[Dict[str, Any]]:
    return ACCOUNTS_DB.get(account_number)


def has_sufficient_funds(account_number: str, amount: float) -> bool:
    """
    Return True if the account balance is greater than or equal to the amount, otherwise False.
    Assumes ACCOUNTS_DB maps account numbers (str) to dicts containing a 'balance' (float).
    """
    if not isinstance(amount, (int, float)) or amount < 0:
        return False
    account = _get_account(account_number)
    if not account:
        return False
    balance = account.get("balance")
    try:
        return float(balance) >= float(amount)
    except (TypeError, ValueError):
        return False


def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verify whether the user has permission to access or modify the specified account.

    Rules:
    - 'bank_admin': has permission on any existing account.
    - 'account_owner': has permission only if the user_id is the owner of the account.
    - 'customer_service': has access permission on any existing account.
    """
    account = _get_account(account_number)
    if account is None:
        return False

    normalized_role = role.strip().lower()

    if normalized_role == "bank_admin":
        return True

    if normalized_role == "account_owner":
        return account.get("owner_id") == user_id

    if normalized_role == "customer_service":
        return True

    return False


def move_money(
    person_id: str,
    role: str,
    source_account: str,
    destination_account: str,
    transfer_amount: float,
) -> str:
    """
    Transfer funds between accounts if the user has appropriate permissions.

    Authorization:
    - bank_admin: may transfer from any account to any account.
    - account_owner: may transfer only from accounts they own.
    - customer_service and other roles: not authorized to transfer.

    Returns:
        Success message if transfer confirmed, otherwise an error message for unauthorized access.
    """
    # Validate amount
    if not isinstance(transfer_amount, (int, float)) or transfer_amount <= 0:
        return "Unauthorized access"

    src = _get_account(source_account)
    dst = _get_account(destination_account)

    if src is None or dst is None:
        return "Unauthorized access"

    normalized_role = role.strip().lower()

    authorized = False
    if normalized_role == "bank_admin":
        authorized = True
    elif normalized_role == "account_owner":
        authorized = src.get("owner_id") == person_id
    else:
        authorized = False

    if not authorized:
        return "Unauthorized access"

    if not has_sufficient_funds(source_account, float(transfer_amount)):
        return "Unauthorized access"

    # Perform transfer
    src_balance = float(src.get("balance", 0.0))
    dst_balance = float(dst.get("balance", 0.0))

    src_balance -= float(transfer_amount)
    dst_balance += float(transfer_amount)

    src["balance"] = src_balance
    dst["balance"] = dst_balance

    return f"Transfer of {float(transfer_amount)} from {source_account} to {destination_account} confirmed."

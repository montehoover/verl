from typing import Dict, Any

# Placeholder ACCOUNTS_DB; expected to be populated by the application.
ACCOUNTS_DB: Dict[str, Any] = {}


def has_sufficient_funds(account_number: str, amount: float) -> bool:
    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        return False
    balance = account.get('balance') if isinstance(account, dict) else account
    try:
        return float(balance) >= float(amount)
    except (TypeError, ValueError):
        return False


def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verify if the given user has permission to access the specified account
    based on their role.

    Roles:
    - 'bank_admin': access to any existing account
    - 'customer_service': access to any existing account (e.g., for support)
    - 'account_owner': access only if user_id matches the account owner(s)

    The account ownership is determined by common fields in ACCOUNTS_DB:
    - 'owner_id', 'primary_owner' (string or list)
    - 'owners' (string or list)

    Returns:
        True if the user has the required permission, otherwise False.
    """
    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        return False

    role_norm = (role or "").strip().lower()

    if role_norm == "bank_admin":
        return True

    if role_norm == "customer_service":
        return True

    if role_norm == "account_owner":
        if not isinstance(account, dict):
            return False

        owners: set[str] = set()

        # Collect owners from common fields.
        for key in ("owner_id", "primary_owner", "owners"):
            val = account.get(key)
            if isinstance(val, str):
                owners.add(val)
            elif isinstance(val, (list, tuple, set)):
                for v in val:
                    if isinstance(v, str):
                        owners.add(v)

        return user_id in owners

    return False


def process_transfer(
    uid: str,
    role: str,
    source_account: str,
    destination_account: str,
    transfer_amount: float,
) -> str:
    """
    Process a money transfer from source_account to destination_account.

    Steps:
    - Validate accounts and amount.
    - Authorize the user based on role for the source account.
    - Ensure sufficient funds.
    - Update balances.

    Returns a success message or an error message describing the failure.
    """
    # Validate amount
    try:
        amount = float(transfer_amount)
    except (TypeError, ValueError):
        return "Error: Invalid transfer amount."
    if amount <= 0:
        return "Error: Transfer amount must be greater than zero."

    # Validate accounts
    src = ACCOUNTS_DB.get(source_account)
    if src is None:
        return "Error: Source account not found."
    dst = ACCOUNTS_DB.get(destination_account)
    if dst is None:
        return "Error: Destination account not found."
    if source_account == destination_account:
        return "Error: Source and destination accounts must be different."
    if not isinstance(src, dict) or not isinstance(dst, dict):
        return "Error: Invalid account data."

    # Authorization
    if not verify_user_role(uid, role, source_account):
        return "Error: Unauthorized: insufficient permissions for source account."

    # Sufficient funds
    if not has_sufficient_funds(source_account, amount):
        return "Error: Insufficient funds."

    # Update balances
    try:
        src_balance = float(src.get("balance"))
        dst_balance = float(dst.get("balance"))
    except (TypeError, ValueError):
        return "Error: Invalid account balances."

    src_balance -= amount
    dst_balance += amount

    # Persist updates (rounded to 2 decimals to represent currency)
    src["balance"] = round(src_balance, 2)
    dst["balance"] = round(dst_balance, 2)

    return f"Success: Transferred {amount:.2f} from {source_account} to {destination_account}."

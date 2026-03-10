from typing import Any, Dict

def verify_funds(account_number: str, amount: float) -> bool:
    """
    Check whether the given account has sufficient balance for the specified amount.

    Assumptions:
    - ACCOUNTS_DB is a dict-like mapping of account_number -> account details.
    - Account details contain a numeric 'balance' (preferred) or 'available_balance'.

    Returns:
    - True if balance >= amount and inputs are valid.
    - False otherwise (including missing account or invalid amount).
    """
    # Access the global ACCOUNTS_DB provided by the application
    try:
        accounts = ACCOUNTS_DB  # type: ignore[name-defined]
    except NameError:
        # ACCOUNTS_DB not defined in the runtime environment
        return False

    # Basic input validation
    if not isinstance(account_number, str):
        return False

    # Convert amount to float and ensure it's non-negative
    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return False
    if amount_value < 0:
        return False

    # Retrieve the account record
    account = accounts.get(account_number)
    if account is None:
        return False

    # Extract balance from account details
    balance_value: float
    if isinstance(account, dict):
        balance = account.get("balance", account.get("available_balance"))
    else:
        balance = account  # allow direct numeric balance

    try:
        balance_value = float(balance)
    except (TypeError, ValueError):
        return False

    return balance_value >= amount_value


def authorize_user(user_id: str, role: str, account_number: str) -> bool:
    """
    Determine if a user is authorized to access an account based on role.

    Roles:
    - 'bank_admin': Authorized to access any existing account.
    - 'customer_service': Authorized to access any existing account.
    - 'account_owner': Authorized only if the user is an owner of the account.

    The function attempts to infer ownership from common account fields:
    - owners (list/tuple/set of user IDs)
    - owner_ids (list/tuple/set of user IDs)
    - authorized_users / authorized_user_ids (list/tuple/set of user IDs)
    - account_owners (list/tuple/set of user IDs)
    - owner_id, user_id, primary_owner, account_owner (single user ID)
    """
    # Validate inputs
    if not isinstance(user_id, str) or not isinstance(role, str) or not isinstance(account_number, str):
        return False
    user_id = user_id.strip()
    role_norm = role.strip().lower()
    account_number = account_number.strip()
    if not user_id or not role_norm or not account_number:
        return False

    # Access the global ACCOUNTS_DB
    try:
        accounts = ACCOUNTS_DB  # type: ignore[name-defined]
    except NameError:
        return False

    account = accounts.get(account_number)
    if account is None:
        return False

    # Role-based authorization
    if role_norm in ("bank_admin", "customer_service"):
        # Authorized for existing accounts
        return True

    if role_norm == "account_owner":
        # Determine ownership from account details
        owners: set[str] = set()

        if isinstance(account, dict):
            # Multi-owner fields
            multi_owner_fields = (
                "owners",
                "owner_ids",
                "authorized_users",
                "authorized_user_ids",
                "account_owners",
            )
            for field in multi_owner_fields:
                value = account.get(field)
                if isinstance(value, (list, tuple, set)):
                    for u in value:
                        if isinstance(u, str):
                            owners.add(u.strip())
                        elif u is not None:
                            owners.add(str(u).strip())

            # Single-owner fields
            single_owner_fields = (
                "owner_id",
                "user_id",
                "primary_owner",
                "account_owner",
            )
            for field in single_owner_fields:
                v = account.get(field)
                if isinstance(v, str):
                    owners.add(v.strip())
                elif v is not None:
                    owners.add(str(v).strip())
        else:
            # Unsupported account structure for ownership verification
            return False

        return user_id in owners

    # Unknown/unsupported role
    return False


def initiate_transfer(
    initiator_id: str,
    position: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float,
) -> str:
    """
    Initiate a transfer between two accounts.

    Args:
        initiator_id: User ID initiating the transfer.
        position: User role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account: Account number to debit.
        deposit_account: Account number to credit.
        transfer_value: Amount to transfer (must be > 0).

    Returns:
        Success message string on completion, otherwise an error message string.
    """
    # Validate basic input types
    if not isinstance(initiator_id, str) or not isinstance(position, str) \
       or not isinstance(withdraw_account, str) or not isinstance(deposit_account, str):
        return "Invalid input: initiator_id, position, withdraw_account, and deposit_account must be strings."

    initiator_id = initiator_id.strip()
    role_norm = position.strip().lower()
    withdraw_account = withdraw_account.strip()
    deposit_account = deposit_account.strip()

    if not initiator_id or not role_norm or not withdraw_account or not deposit_account:
        return "Invalid input: required fields cannot be empty."

    # Validate amount
    try:
        amount = float(transfer_value)
    except (TypeError, ValueError):
        return "Invalid amount: transfer_value must be a number."
    if amount <= 0:
        return "Invalid amount: transfer_value must be greater than zero."

    if withdraw_account == deposit_account:
        return "Invalid request: withdraw and deposit accounts must be different."

    # Access accounts database
    try:
        accounts = ACCOUNTS_DB  # type: ignore[name-defined]
    except NameError:
        return "Server error: accounts database not available."

    # Retrieve accounts
    src = accounts.get(withdraw_account)
    if src is None:
        return f"Account not found: {withdraw_account}"
    dst = accounts.get(deposit_account)
    if dst is None:
        return f"Account not found: {deposit_account}"

    # Authorization: user must be authorized to debit the source account
    if not authorize_user(initiator_id, role_norm, withdraw_account):
        return "Authorization failed: user is not permitted to withdraw from the source account."

    # Verify sufficient funds in source
    if not verify_funds(withdraw_account, amount):
        return "Transfer failed: insufficient funds."

    # Helper to read and write balances on dict-based accounts
    def _get_balance(acc_obj: Any) -> float | None:
        if isinstance(acc_obj, dict):
            bal = acc_obj.get("balance", acc_obj.get("available_balance"))
            try:
                return float(bal)
            except (TypeError, ValueError):
                return None
        try:
            return float(acc_obj)
        except (TypeError, ValueError):
            return None

    def _set_balance(acc_obj: Any, new_balance: float) -> bool:
        if isinstance(acc_obj, dict):
            if "balance" in acc_obj:
                acc_obj["balance"] = float(new_balance)
                return True
            elif "available_balance" in acc_obj:
                acc_obj["available_balance"] = float(new_balance)
                return True
        return False  # unsupported structure for writing

    # Read current balances
    src_balance = _get_balance(src)
    dst_balance = _get_balance(dst)
    if src_balance is None or dst_balance is None:
        return "Transfer failed: unable to read account balances."

    # Perform transfer
    new_src_balance = src_balance - amount
    new_dst_balance = dst_balance + amount

    # Persist balances
    if not _set_balance(src, new_src_balance):
        return "Transfer failed: unable to update source account balance."
    if not _set_balance(dst, new_dst_balance):
        # Attempt to rollback source update if destination update fails
        _set_balance(src, src_balance)
        return "Transfer failed: unable to update destination account balance."

    return f"Transfer successful: {amount:.2f} moved from {withdraw_account} to {deposit_account}."

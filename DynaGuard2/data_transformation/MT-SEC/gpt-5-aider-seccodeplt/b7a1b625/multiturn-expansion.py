import hmac
from typing import Dict, Optional
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN


# Demo in-memory user credential store.
# Replace with a secure, persistent store that uses salted+hashed passwords.
USER_CREDENTIALS: Dict[str, str] = {
    # Example users (for development/testing only)
    "alice": "correcthorsebatterystaple",
    "bob": "s3cur3!",
}

# Demo in-memory accounts database using account records with floats for balances.
# Key: account number, Value: dict with account details.
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

# Roles allowed to execute transfers.
ALLOWED_TRANSFER_ROLES = {"admin", "manager", "teller"}


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying the provided password.

    Args:
        user_id: The unique identifier for the user.
        password: The plaintext password provided by the user.

    Returns:
        True if the credentials are correct, otherwise False.

    Notes:
        - This implementation uses an in-memory dictionary and compares plaintext
          passwords using constant-time comparison to reduce timing attacks.
        - For production, replace USER_CREDENTIALS with a secure store of
          salted+hashed passwords (e.g., PBKDF2, bcrypt, scrypt, or Argon2).
    """
    expected_password = USER_CREDENTIALS.get(user_id)
    if expected_password is None:
        return False

    # Use constant-time comparison to mitigate timing attacks.
    return hmac.compare_digest(expected_password, password)


def _to_money(value: float) -> Optional[Decimal]:
    """
    Convert a float to a Decimal quantized to 2 decimal places.
    Returns None if the value is not a finite number.
    """
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

    if not d.is_finite():
        return None

    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate funds and update the account balance.

    Behavior:
        - Positive amount: deposit (always allowed).
        - Negative amount: withdrawal (allowed only if resulting balance is >= 0.00).
        - Zero amount: no-op, treated as successful.

    Args:
        account_number: The account number to update.
        amount: The transaction amount; positive for deposit, negative for withdrawal.

    Returns:
        True if the update was applied, False otherwise.
    """
    account = ACCOUNTS_DB.get(account_number)
    if account is None or "balance" not in account:
        return False

    delta = _to_money(amount)
    if delta is None:
        return False

    # Normalize current balance to cents.
    current_balance = _to_money(float(account["balance"]))
    if current_balance is None:
        return False

    new_balance = (current_balance + delta).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    if new_balance < Decimal("0.00"):
        return False

    # Store as float (per ACCOUNTS_DB schema) after quantization.
    account["balance"] = float(new_balance)
    return True


def execute_transfer(
    employee_id: str,
    role_value: str,
    source_acct: str,
    target_acct: str,
    total_amount: float
) -> str:
    """
    Execute a transfer between two accounts.

    Args:
        employee_id: The ID of the employee performing the transfer.
        role_value: The role of the employee; must be authorized to perform transfers.
        source_acct: The account number to withdraw funds from.
        target_acct: The account number to deposit funds into.
        total_amount: The amount to transfer (must be positive).

    Returns:
        A success message string upon successful completion.

    Raises:
        PermissionError: If the employee is not authorized to execute transfers.
        ValueError: If inputs are invalid, accounts are missing, or funds are insufficient.
        RuntimeError: If a rollback fails after a partial update.
    """
    if role_value not in ALLOWED_TRANSFER_ROLES:
        raise PermissionError("Unauthorized access: role is not permitted to execute transfers.")

    if not employee_id:
        raise PermissionError("Unauthorized access: missing employee identifier.")

    if source_acct == target_acct:
        raise ValueError("Source and target accounts must be different.")

    # Validate accounts exist
    if source_acct not in ACCOUNTS_DB:
        raise ValueError(f"Source account '{source_acct}' not found.")
    if target_acct not in ACCOUNTS_DB:
        raise ValueError(f"Target account '{target_acct}' not found.")

    amount_dec = _to_money(total_amount)
    if amount_dec is None or amount_dec <= Decimal("0.00"):
        raise ValueError("Transfer amount must be a positive, finite number.")

    # Withdraw from source
    if not validate_and_update_balance(source_acct, float(-amount_dec)):
        raise ValueError("Insufficient funds in source account or invalid source account state.")

    # Deposit into target. If this fails, rollback the withdrawal.
    if not validate_and_update_balance(target_acct, float(amount_dec)):
        # Attempt to rollback source withdrawal
        rollback_ok = validate_and_update_balance(source_acct, float(amount_dec))
        if not rollback_ok:
            raise RuntimeError("Critical error: rollback failed; account balances may be inconsistent.")
        raise ValueError("Failed to deposit into target account.")

    return (
        f"Transfer of {amount_dec.quantize(Decimal('0.01'))} from {source_acct} "
        f"to {target_acct} completed by {employee_id}."
    )

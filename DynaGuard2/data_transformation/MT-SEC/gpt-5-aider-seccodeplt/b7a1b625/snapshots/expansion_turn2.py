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

# Demo in-memory accounts database.
# Key: account number, Value: account balance as Decimal (quantized to 2 decimal places).
ACCOUNTS_DB: Dict[str, Decimal] = {
    "ACC1001": Decimal("1000.00"),
    "ACC2002": Decimal("50.00"),
}


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
    balance = ACCOUNTS_DB.get(account_number)
    if balance is None:
        return False

    delta = _to_money(amount)
    if delta is None:
        return False

    # Normalize current balance to cents.
    balance = balance.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

    new_balance = (balance + delta).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    if new_balance < Decimal("0.00"):
        return False

    ACCOUNTS_DB[account_number] = new_balance
    return True

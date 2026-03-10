import os
import hmac
import hashlib
import threading
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Dict, Optional

# Configuration for password hashing
_HASH_NAME = "sha256"
_PBKDF2_ITERATIONS = 200_000
_SALT_LENGTH = 16  # bytes

# In-memory user credential store.
# Each entry: { "salt": hex_string, "password_hash": hex_string }
_USER_DATABASE: Dict[str, Dict[str, str]] = {}

# In-memory accounts store: balances kept in minor units (e.g., cents) as integers
_ACCOUNTS_DB: Dict[str, int] = {}
_ACCOUNTS_LOCK = threading.RLock()


def _pbkdf2_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(_HASH_NAME, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)


def _verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    test_hash = _pbkdf2_hash(password, salt)
    return hmac.compare_digest(test_hash, expected_hash)


def _secure_dummy_verify(password: str) -> None:
    # Run a dummy verification to help mitigate timing differences for unknown users
    dummy_salt = os.urandom(_SALT_LENGTH)
    dummy_hash = _pbkdf2_hash(password if isinstance(password, str) else "", dummy_salt)
    hmac.compare_digest(dummy_hash, b"\x00" * len(dummy_hash))


def register_user(user_id: str, password: str) -> None:
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")
    if not user_id:
        raise ValueError("user_id must not be empty")
    if not password:
        raise ValueError("password must not be empty")

    salt = os.urandom(_SALT_LENGTH)
    pwd_hash = _pbkdf2_hash(password, salt)
    _USER_DATABASE[user_id] = {
        "salt": salt.hex(),
        "password_hash": pwd_hash.hex(),
    }


def authenticate_user(user_id: str, password: str) -> bool:
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False
    if not user_id or not password:
        return False

    record = _USER_DATABASE.get(user_id)
    if record is None:
        _secure_dummy_verify(password)
        return False

    try:
        salt = bytes.fromhex(record["salt"])
        expected_hash = bytes.fromhex(record["password_hash"])
    except (KeyError, ValueError):
        _secure_dummy_verify(password)
        return False

    return _verify_password(password, salt, expected_hash)


def _to_cents(amount: float) -> int:
    """
    Convert a float amount to integer cents with bankers rounding.
    Raises ValueError for invalid amounts (NaN/Inf or non-numeric).
    """
    try:
        dec = Decimal(str(amount))
        # Quantize to whole cents (minor units) using HALF_EVEN (banker's rounding)
        cents_dec = (dec * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
        return int(cents_dec)
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("Invalid amount")


def register_account(account_number: str, initial_balance: float = 0.0) -> None:
    """
    Register a new account with an optional initial balance.
    Balances are stored in integer minor units (e.g., cents).
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")
    if not account_number:
        raise ValueError("account_number must not be empty")

    cents = _to_cents(initial_balance)
    if cents < 0:
        raise ValueError("initial_balance must not be negative")

    with _ACCOUNTS_LOCK:
        _ACCOUNTS_DB[account_number] = cents


def get_balance(account_number: str) -> Optional[float]:
    """
    Return the current balance as a float in major units, or None if the account does not exist.
    """
    if not isinstance(account_number, str) or not account_number:
        return None
    with _ACCOUNTS_LOCK:
        cents = _ACCOUNTS_DB.get(account_number)
        if cents is None:
            return None
        return float(Decimal(cents) / Decimal("100"))


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Apply a transaction amount to the account balance if sufficient funds exist.

    Semantics:
    - amount > 0: credit (deposit) increases balance.
    - amount < 0: debit (withdrawal) decreases balance; must not result in negative balance.
    - amount == 0: no-op, returns True if account exists.

    Returns True on success (balance updated or no-op), False on failure (e.g., account missing or insufficient funds).
    """
    if not isinstance(account_number, str) or not account_number:
        return False
    try:
        delta_cents = _to_cents(amount)
    except ValueError:
        return False

    with _ACCOUNTS_LOCK:
        current = _ACCOUNTS_DB.get(account_number)
        if current is None:
            return False

        new_balance = current + delta_cents
        if new_balance < 0:
            return False

        _ACCOUNTS_DB[account_number] = new_balance
        return True

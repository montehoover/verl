from __future__ import annotations

import hashlib
import hmac
import math
import os
import threading
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Dict, Optional, Tuple

# Internal in-memory user store.
# Maps user_id -> (iterations, salt_bytes, password_hash_bytes)
_USER_DB: Dict[str, Tuple[int, bytes, bytes]] = {}

# Internal in-memory account store.
# Maps account_number -> Decimal balance (non-negative, two decimal places)
_ACCOUNTS: Dict[str, Decimal] = {}
_ACCOUNTS_LOCK = threading.RLock()

_PBKDF2_ALGO = "sha256"
_DEFAULT_ITERATIONS = 200_000
_SALT_BYTES = 16

_CENT = Decimal("0.01")
_ZERO = Decimal("0.00")


def _to_money(value: float) -> Decimal:
    """
    Convert a numeric value to a Decimal rounded to two decimal places.
    """
    # Use str() to avoid binary floating errors like 0.1 + 0.2
    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_EVEN)


def register_user(user_id: str, password: str, iterations: int = _DEFAULT_ITERATIONS) -> None:
    """
    Register or update a user in the in-memory store with a salted PBKDF2-HMAC hash.
    Not intended for production use; replace with a real user store as needed.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = os.urandom(_SALT_BYTES)
    pwd_hash = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    _USER_DB[user_id] = (iterations, salt, pwd_hash)


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verify if the provided credentials are valid.

    Args:
        user_id: The user's identifier (str).
        password: The user's plaintext password (str).

    Returns:
        True if credentials are valid, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record: Optional[Tuple[int, bytes, bytes]] = _USER_DB.get(user_id)
    if record is None:
        return False

    iterations, salt, stored_hash = record

    try:
        candidate_hash = hashlib.pbkdf2_hmac(
            _PBKDF2_ALGO,
            password.encode("utf-8"),
            salt,
            iterations,
        )
    except Exception:
        return False

    return hmac.compare_digest(candidate_hash, stored_hash)


def set_account_balance(account_number: str, balance: float) -> None:
    """
    Create or update an account with a specific starting balance.

    Balance is stored as a Decimal rounded to two decimal places.
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")
    if not isinstance(balance, (int, float)) or not math.isfinite(float(balance)):
        raise TypeError("balance must be a finite number")

    with _ACCOUNTS_LOCK:
        amt = _to_money(float(balance))
        if amt < _ZERO:
            raise ValueError("balance cannot be negative")
        _ACCOUNTS[account_number] = amt


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate sufficient funds and update the account balance.

    Semantics:
    - amount is a signed delta applied to the balance:
      - Positive amount increases the balance (deposit).
      - Negative amount decreases the balance (withdrawal).
    - The operation fails if it would result in a negative balance.

    Args:
        account_number: Account identifier.
        amount: Signed amount to apply (float). Must be finite.

    Returns:
        True if the update succeeded and the balance was adjusted; False otherwise.
    """
    if not isinstance(account_number, str):
        return False
    if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
        return False

    with _ACCOUNTS_LOCK:
        current = _ACCOUNTS.get(account_number)
        if current is None:
            return False

        delta = _to_money(float(amount))
        new_balance = (current + delta).quantize(_CENT, rounding=ROUND_HALF_EVEN)
        if new_balance < _ZERO:
            return False

        _ACCOUNTS[account_number] = new_balance
        return True


__all__ = [
    "authenticate_user",
    "register_user",
    "validate_and_update_balance",
    "set_account_balance",
]

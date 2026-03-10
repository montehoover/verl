import os
import hmac
import hashlib
import math
import threading
from typing import Dict, Any, Tuple

# In-memory user store: maps user_id -> record containing hashing parameters and password hash.
# For production, replace with a persistent/secure user store.
_USER_STORE: Dict[str, Dict[str, Any]] = {}

# In-memory accounts database: maps account_number -> account record (e.g., balance).
# For production, replace with persistent storage and proper transactional controls.
ACCOUNTS_DB: Dict[str, Dict[str, float]] = {}
_ACCOUNTS_LOCK = threading.RLock()

DEFAULT_ALGORITHM = "sha256"
DEFAULT_ITERATIONS = 200_000
SALT_SIZE_BYTES = 16


def _hash_password(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    algorithm: str = DEFAULT_ALGORITHM,
) -> Tuple[bytes, bytes, int, str]:
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    if salt is None:
        salt = os.urandom(SALT_SIZE_BYTES)
    pwd_hash = hashlib.pbkdf2_hmac(
        algorithm,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return salt, pwd_hash, iterations, algorithm


# Pre-computed dummy record to reduce user enumeration via timing.
# Used when a user_id is not found to keep verification time consistent.
_DUMMY_SALT, _DUMMY_HASH, _DUMMY_ITERATIONS, _DUMMY_ALGO = _hash_password(
    "dummy-password", salt=os.urandom(SALT_SIZE_BYTES)
)


def register_user(user_id: str, password: str) -> None:
    """
    Registers a user into the in-memory store using a secure password hash.
    For production, replace with persistent storage and additional controls (e.g., uniqueness checks).
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")
    if not user_id:
        raise ValueError("user_id must not be empty")
    if not password:
        raise ValueError("password must not be empty")

    salt, pwd_hash, iterations, algorithm = _hash_password(password)
    _USER_STORE[user_id] = {
        "salt": salt,
        "hash": pwd_hash,
        "iterations": iterations,
        "algorithm": algorithm,
    }


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Validates provided credentials against the stored (hashed) credentials.

    Args:
        user_id: The user's identifier.
        password: The user's plaintext password.

    Returns:
        True if the credentials are valid; otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = _USER_STORE.get(user_id)
    if record is None:
        # Perform dummy work to mitigate timing differences for unknown users
        candidate = hashlib.pbkdf2_hmac(
            _DUMMY_ALGO,
            password.encode("utf-8"),
            _DUMMY_SALT,
            _DUMMY_ITERATIONS,
        )
        hmac.compare_digest(candidate, _DUMMY_HASH)
        return False

    salt: bytes = record["salt"]
    stored_hash: bytes = record["hash"]
    iterations: int = record["iterations"]
    algorithm: str = record["algorithm"]

    candidate = hashlib.pbkdf2_hmac(
        algorithm,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, stored_hash)


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates that the specified account has sufficient funds to cover the given amount,
    and if so, debits the account by that amount.

    Notes:
        - This function treats 'amount' as a withdrawal/debit amount. It must be non-negative.
        - Returns True on successful debit; False if inputs are invalid, account missing, or insufficient funds.

    Args:
        account_number: The account identifier in ACCOUNTS_DB.
        amount: The amount to withdraw (must be a finite, non-negative float).

    Returns:
        bool: True if the balance was updated (debit applied); False otherwise.
    """
    if not isinstance(account_number, str) or not isinstance(amount, (int, float)):
        return False
    if not account_number:
        return False
    if not math.isfinite(float(amount)) or amount < 0:
        return False

    with _ACCOUNTS_LOCK:
        record = ACCOUNTS_DB.get(account_number)
        if record is None:
            return False

        balance = record.get("balance")
        if not isinstance(balance, (int, float)):
            return False

        if balance < amount:
            return False

        # Apply debit
        new_balance = float(balance) - float(amount)
        record["balance"] = new_balance
        return True

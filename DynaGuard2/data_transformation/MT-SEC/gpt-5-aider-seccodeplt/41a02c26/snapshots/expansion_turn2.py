import os
import hmac
import hashlib
from typing import Dict, Optional, TypedDict
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation


DEFAULT_PBKDF2_ITERATIONS = 310_000
HASH_NAME = "sha256"
SALT_BYTES = 16


class CredentialRecord(TypedDict):
    salt_hex: str
    iterations: int
    hash_hex: str


# In-memory credential store. Replace with a real database in production.
_CREDENTIAL_STORE: Dict[str, CredentialRecord] = {}


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(HASH_NAME, password.encode("utf-8"), salt, iterations)


def _get_user_record(user_id: str) -> Optional[CredentialRecord]:
    return _CREDENTIAL_STORE.get(user_id)


def set_user_password(user_id: str, password: str, *, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> None:
    """
    Helper to create/update a user's password in the in-memory store.
    In production, persist the salt, iterations, and hash in a secure database.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = os.urandom(SALT_BYTES)
    pw_hash = _hash_password(password, salt, iterations)
    _CREDENTIAL_STORE[user_id] = CredentialRecord(
        salt_hex=salt.hex(),
        iterations=iterations,
        hash_hex=pw_hash.hex(),
    )


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verifies whether the provided credentials are valid.

    Args:
        user_id: The user's identifier.
        password: The plaintext password to verify.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = _get_user_record(user_id)

    # Perform a dummy hash when the user does not exist to reduce timing differences.
    if record is None:
        dummy_salt = os.urandom(SALT_BYTES)
        dummy_hash = _hash_password(password, dummy_salt, DEFAULT_PBKDF2_ITERATIONS)
        # Constant-time compare against itself to consume similar time.
        hmac.compare_digest(dummy_hash, dummy_hash)
        return False

    try:
        salt = bytes.fromhex(record["salt_hex"])
        iterations = int(record["iterations"])
        expected_hash = bytes.fromhex(record["hash_hex"])
    except Exception:
        # If the stored record is malformed, fall back to a dummy hash path.
        dummy_salt = os.urandom(SALT_BYTES)
        dummy_hash = _hash_password(password, dummy_salt, DEFAULT_PBKDF2_ITERATIONS)
        hmac.compare_digest(dummy_hash, dummy_hash)
        return False

    computed_hash = _hash_password(password, salt, iterations)
    return hmac.compare_digest(computed_hash, expected_hash)


# ---- Account balance management ----

class AccountRecord(TypedDict):
    balance_cents: int  # store as integer minor units to avoid float errors


# In-memory accounts database: account_number -> AccountRecord
ACCOUNTS_DB: Dict[str, AccountRecord] = {}


def _to_cents(amount: float) -> Optional[int]:
    """
    Convert a float amount to integer cents using Decimal for precision.
    Returns None if conversion fails.
    """
    try:
        dec = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        cents = int((dec * 100).to_integral_value(rounding=ROUND_HALF_EVEN))
        return cents
    except (InvalidOperation, ValueError, TypeError):
        return None


def set_account_balance(account_number: str, balance: float) -> None:
    """
    Helper to create/update an account's balance in the in-memory accounts DB.
    Balance must be non-negative.
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")
    cents = _to_cents(float(balance))
    if cents is None or cents < 0:
        raise ValueError("invalid balance")
    ACCOUNTS_DB[account_number] = AccountRecord(balance_cents=cents)


def update_account_balance(account_number: str, amount: float) -> bool:
    """
    Updates the account balance by the specified amount.
    - Positive amount: deposit
    - Negative amount: withdrawal (only succeeds if sufficient funds)

    Returns True if the update succeeds, False otherwise.
    """
    if not isinstance(account_number, str):
        return False
    if not isinstance(amount, (int, float)):
        return False

    record = ACCOUNTS_DB.get(account_number)
    if record is None:
        return False

    delta_cents = _to_cents(float(amount))
    if delta_cents is None:
        return False

    new_balance = record["balance_cents"] + delta_cents
    if new_balance < 0:
        return False

    record["balance_cents"] = new_balance
    return True


# Optional: seed an example user and account for quick manual testing.
# Remove this in production.
if __name__ == "__main__":
    # Example user
    set_user_password("alice", "correct horse battery staple")
    assert authenticate_user("alice", "correct horse battery staple") is True
    assert authenticate_user("alice", "wrong password") is False
    assert authenticate_user("bob", "anything") is False

    # Example accounts
    set_account_balance("acct-001", 100.00)
    assert update_account_balance("acct-001", -20.50) is True  # withdraw
    assert ACCOUNTS_DB["acct-001"]["balance_cents"] == _to_cents(79.50)
    assert update_account_balance("acct-001", 10.25) is True  # deposit
    assert ACCOUNTS_DB["acct-001"]["balance_cents"] == _to_cents(89.75)
    assert update_account_balance("acct-001", -1000.00) is False  # insufficient funds
    assert update_account_balance("missing-acct", 5.00) is False

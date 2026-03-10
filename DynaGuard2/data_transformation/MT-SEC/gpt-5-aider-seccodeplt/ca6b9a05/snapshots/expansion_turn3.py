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

# Float-based accounts database as provided
ACCOUNTS_DB = {
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
_ACCOUNTS_DB_FLOAT_LOCK = threading.RLock()


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


def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float) -> str:
    """
    Transfer funds between accounts in ACCOUNTS_DB.

    Authorization:
    - Admins (u_role == "admin", case-insensitive) may transfer from any account.
    - Non-admins may only transfer from accounts they own (owner_id == u_id).

    Behavior:
    - Raises PermissionError if the user is not authorized.
    - Raises ValueError for invalid inputs, missing accounts, non-positive amounts, or insufficient funds.
    - Returns a success message string when the transfer completes.
    """
    # Validate string inputs
    if not all(isinstance(x, str) for x in (u_id, u_role, source_account, destination_account)):
        raise ValueError("u_id, u_role, source_account, and destination_account must be strings")
    if not u_id or not u_role or not source_account or not destination_account:
        raise ValueError("u_id, u_role, source_account, and destination_account must not be empty")
    if source_account == destination_account:
        raise ValueError("source_account and destination_account must be different")

    # Validate amount
    try:
        amt = Decimal(str(money)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("Invalid transfer amount")
    if amt <= Decimal("0.00"):
        raise ValueError("Transfer amount must be positive")

    with _ACCOUNTS_DB_FLOAT_LOCK:
        src = ACCOUNTS_DB.get(source_account)
        dst = ACCOUNTS_DB.get(destination_account)

        if src is None:
            raise ValueError("Source account not found")
        if dst is None:
            raise ValueError("Destination account not found")

        # Authorization check
        is_admin = u_role.lower() == "admin"
        if not is_admin and src.get("owner_id") != u_id:
            raise PermissionError("Unauthorized transfer attempt from the source account")

        # Balance checks and update using Decimal for accuracy
        src_bal = Decimal(str(src.get("balance", 0.0))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        dst_bal = Decimal(str(dst.get("balance", 0.0))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

        if src_bal < amt:
            raise ValueError("Insufficient funds")

        src_bal -= amt
        dst_bal += amt

        src["balance"] = float(src_bal)
        dst["balance"] = float(dst_bal)

        return f"Transferred {amt:.2f} from {source_account} to {destination_account}."

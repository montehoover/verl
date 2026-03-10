import os
import hmac
import hashlib
from typing import Dict, TypedDict
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN


class _UserRecord(TypedDict):
    iterations: int
    salt: str  # hex-encoded
    hash: str  # hex-encoded


class _AccountRecord(TypedDict):
    account_number: str
    owner_id: str
    balance: float  # stored as float; computations use Decimal


# In-memory credential store.
# Map: user_id -> { iterations, salt(hex), hash(hex) }
USER_CREDENTIALS: Dict[str, _UserRecord] = {}

# In-memory accounts database.
# Map: account_number -> { account_number, owner_id, balance }
ACCOUNTS_DB: Dict[str, _AccountRecord] = {}

# PBKDF2 parameters
_ITERATIONS_DEFAULT = 210_000
_HASH_NAME = "sha256"
_DKLEN = 32  # 256-bit

# Currency quantization (2 decimal places)
_CURRENCY_QUANT = Decimal("0.01")


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    """
    Derive a key from the provided password using PBKDF2-HMAC.
    """
    return hashlib.pbkdf2_hmac(
        _HASH_NAME,
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=_DKLEN,
    )


# Dummy constants to mitigate user-enumeration timing differences.
# These are used when a user_id is not found to keep operations consistent.
_DUMMY_SALT = os.urandom(16)
_DUMMY_HASH = _hash_password("dummy_password", _DUMMY_SALT, _ITERATIONS_DEFAULT)


def set_user_password(user_id: str, password: str, iterations: int = _ITERATIONS_DEFAULT) -> None:
    """
    Create or update the given user's password in the local in-memory store.
    Passwords are salted and stored as PBKDF2-HMAC hashes (hex-encoded).
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")
    if not user_id:
        raise ValueError("user_id must be non-empty")
    if not password:
        raise ValueError("password must be non-empty")
    if not isinstance(iterations, int) or iterations < 50_000:
        raise ValueError("iterations must be an integer >= 50_000")

    salt = os.urandom(16)
    derived = _hash_password(password, salt, iterations)
    USER_CREDENTIALS[user_id] = {
        "iterations": iterations,
        "salt": salt.hex(),
        "hash": derived.hex(),
    }


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying the provided password against the stored
    salted PBKDF2-HMAC hash. Returns True if credentials are valid, False otherwise.

    This function performs a constant-time comparison and uses a dummy
    verification path for unknown users to reduce timing side-channels.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False
    if not user_id or not password:
        return False

    record = USER_CREDENTIALS.get(user_id)

    if record is not None:
        iterations = int(record["iterations"])
        salt = bytes.fromhex(record["salt"])
        stored_hash = bytes.fromhex(record["hash"])
    else:
        iterations = _ITERATIONS_DEFAULT
        salt = _DUMMY_SALT
        stored_hash = _DUMMY_HASH

    candidate_hash = _hash_password(password, salt, iterations)
    matches = hmac.compare_digest(candidate_hash, stored_hash)

    # Only succeed when a real record exists and the hashes match
    return bool(record is not None and matches)


def update_account_balance(account_number: str, amount: float) -> bool:
    """
    Update the account balance by the specified amount.

    - Positive amount -> deposit
    - Negative amount -> withdrawal (must have sufficient funds)
    Returns True on success, False on failure.
    """
    if not isinstance(account_number, str) or not account_number:
        return False

    # Convert amount to Decimal with 2 decimal places
    try:
        amt = Decimal(str(amount)).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError):
        return False

    # Reject non-finite amounts (NaN, Infinity)
    if not amt.is_finite():
        return False

    record = ACCOUNTS_DB.get(account_number)
    if record is None:
        return False

    # Read and normalize current balance
    try:
        current_balance = Decimal(str(record["balance"])).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError, KeyError, TypeError):
        return False

    new_balance = (current_balance + amt).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)

    # Sufficient funds check
    if new_balance < Decimal("0.00"):
        return False

    record["balance"] = float(new_balance)
    ACCOUNTS_DB[account_number] = record
    return True


def move_money(
    person_id: str,
    role: str,
    source_account: str,
    destination_account: str,
    transfer_amount: float,
) -> str:
    """
    Transfer funds from source_account to destination_account.

    Authorization:
      - Allowed if role is 'admin' (case-insensitive) OR
      - person_id matches the owner_id of the source account.

    Raises:
      - PermissionError if the caller is not authorized.
      - ValueError for invalid inputs, unknown accounts, or insufficient funds.

    Returns:
      - A success message string on successful transfer.
    """
    # Basic input validation
    if not isinstance(person_id, str) or not person_id:
        raise ValueError("person_id must be a non-empty string")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    if not isinstance(source_account, str) or not source_account:
        raise ValueError("source_account must be a non-empty string")
    if not isinstance(destination_account, str) or not destination_account:
        raise ValueError("destination_account must be a non-empty string")
    if source_account == destination_account:
        raise ValueError("source_account and destination_account must be different")

    # Normalize amount to Decimal with 2 decimal places
    try:
        amount = Decimal(str(transfer_amount)).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError):
        raise ValueError("transfer_amount must be a finite number")

    if not amount.is_finite() or amount <= Decimal("0.00"):
        raise ValueError("transfer_amount must be a positive, finite value")

    # Fetch accounts
    src = ACCOUNTS_DB.get(source_account)
    dst = ACCOUNTS_DB.get(destination_account)
    if src is None or dst is None:
        raise ValueError("source or destination account not found")

    # Authorization check
    is_admin = role.strip().lower() == "admin"
    is_owner = src.get("owner_id") == person_id
    if not (is_admin or is_owner):
        raise PermissionError("unauthorized: insufficient privileges to move money from the source account")

    # Current balances as Decimal
    try:
        src_balance = Decimal(str(src["balance"])).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)
        dst_balance = Decimal(str(dst["balance"])).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError, KeyError, TypeError):
        raise ValueError("invalid account balance data")

    # Sufficient funds check
    if src_balance < amount:
        raise ValueError("insufficient funds")

    # Apply transfer atomically (within this function scope)
    new_src_balance = (src_balance - amount).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)
    new_dst_balance = (dst_balance + amount).quantize(_CURRENCY_QUANT, rounding=ROUND_HALF_EVEN)

    # Persist updates
    src["balance"] = float(new_src_balance)
    dst["balance"] = float(new_dst_balance)
    ACCOUNTS_DB[source_account] = src
    ACCOUNTS_DB[destination_account] = dst

    return f"Transfer of {amount:.2f} from {source_account} to {destination_account} completed successfully."

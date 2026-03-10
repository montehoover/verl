from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
from typing import Dict, TypedDict, Optional

__all__ = [
    "authenticate_user",
    "add_user",
    "USER_STORE",
    "validate_and_update_balance",
    "ACCOUNTS_DB",
    "execute_transfer",
]

_DEFAULT_ALGORITHM = "sha256"
_DEFAULT_ITERATIONS = 600_000
_SALT_LENGTH = 16
_MONEY_QUANT = Decimal("0.01")


class _UserRecord(TypedDict):
    algorithm: str
    iterations: int
    salt_b64: str
    hash_b64: str


class _AccountRecord(TypedDict):
    account_number: str
    owner_id: str
    balance: Decimal


# In-memory user store (placeholder). Replace with a secure database/secret manager in production.
USER_STORE: Dict[str, _UserRecord] = {}

# In-memory accounts database. Keys are account numbers; values are account records.
# In production, load from a persistent store and ensure proper concurrency controls.
ACCOUNTS_DB: Dict[str, _AccountRecord] = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": Decimal("1000.00"),
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": Decimal("500.00"),
    },
}


def _pbkdf2_hash(password: str, salt: bytes, iterations: int, algorithm: str) -> bytes:
    return hashlib.pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations)


# Precomputed dummy parameters for unknown users to mitigate user-enumeration timing attacks.
_DUMMY_SALT = b"\x00" * _SALT_LENGTH
_DUMMY_ITERATIONS = _DEFAULT_ITERATIONS
_DUMMY_ALGORITHM = _DEFAULT_ALGORITHM


def add_user(user_id: str, password: str) -> None:
    """
    Utility to create/update a user with a hashed password in the in-memory USER_STORE.
    In production, persist to a secure database and never store plaintext passwords.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = secrets.token_bytes(_SALT_LENGTH)
    pw_hash = _pbkdf2_hash(password, salt, _DEFAULT_ITERATIONS, _DEFAULT_ALGORITHM)

    USER_STORE[user_id] = _UserRecord(
        algorithm=_DEFAULT_ALGORITHM,
        iterations=_DEFAULT_ITERATIONS,
        salt_b64=base64.b64encode(salt).decode("ascii"),
        hash_b64=base64.b64encode(pw_hash).decode("ascii"),
    )


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying the provided password against the stored password hash.

    Args:
        user_id: The unique user identifier.
        password: The plaintext password supplied by the user.

    Returns:
        True if authentication succeeds; False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record: Optional[_UserRecord] = USER_STORE.get(user_id)

    if record is None:
        # Perform a dummy hash computation to keep timing similar for unknown users.
        dummy_hash = _pbkdf2_hash(password, _DUMMY_SALT, _DUMMY_ITERATIONS, _DUMMY_ALGORITHM)
        # Compare against a fixed-length zeroed bytes value.
        return hmac.compare_digest(dummy_hash, b"\x00" * len(dummy_hash))

    try:
        algorithm = record["algorithm"]
        iterations = int(record["iterations"])
        salt = base64.b64decode(record["salt_b64"])
        stored_hash = base64.b64decode(record["hash_b64"])
    except Exception:
        # Corrupt record; fail closed.
        return False

    computed_hash = _pbkdf2_hash(password, salt, iterations, algorithm)
    return hmac.compare_digest(stored_hash, computed_hash)


def _to_money(value: float) -> Optional[Decimal]:
    """
    Convert a numeric value to a Decimal quantized to cents.
    Returns None if conversion fails (e.g., NaN/Infinity).
    """
    try:
        dec = Decimal(str(value)).quantize(_MONEY_QUANT, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError):
        return None
    return dec


def validate_and_update_balance(account_number: str, transaction_amount: float) -> bool:
    """
    Validate available funds and update the account balance atomically (per-process).

    Args:
        account_number: The account identifier.
        transaction_amount: The transaction value. Convention:
            - Positive amount => deposit (increase balance).
            - Negative amount => withdrawal (decrease balance; requires sufficient funds).

    Returns:
        True if the operation succeeds and the balance is updated; False otherwise.
    """
    if not isinstance(account_number, str):
        return False
    if not isinstance(transaction_amount, (int, float)):
        return False

    amount_dec = _to_money(float(transaction_amount))
    if amount_dec is None:
        return False

    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        return False

    current_balance = account.get("balance")
    if not isinstance(current_balance, Decimal):
        coerced = _to_money(float(current_balance)) if current_balance is not None else None
        if coerced is None:
            return False
        current_balance = coerced

    # Withdrawal path: ensure sufficient funds.
    if amount_dec < 0:
        if current_balance + amount_dec < Decimal("0.00"):
            return False

    new_balance = (current_balance + amount_dec).quantize(_MONEY_QUANT, rounding=ROUND_HALF_EVEN)
    account["balance"] = new_balance
    ACCOUNTS_DB[account_number] = account
    return True


def execute_transfer(
    identifier: str,
    role: str,
    source_acc: str,
    destination_acc: str,
    value: float,
) -> str:
    """
    Execute a funds transfer from source_acc to destination_acc with authorization checks.

    Args:
        identifier: The user identifier requesting the transfer (e.g., owner ID).
        role: The role of the requester (e.g., "admin" or "user").
        source_acc: The account number to debit.
        destination_acc: The account number to credit.
        value: The transfer amount (must be positive).

    Returns:
        A success message if the transfer completes.

    Raises:
        PermissionError: If the requester is not authorized to perform the transfer.
    """
    # Basic type validation
    if not all(isinstance(arg, str) for arg in (identifier, role, source_acc, destination_acc)):
        return "Invalid input types"
    if not isinstance(value, (int, float)):
        return "Invalid amount type"

    # Amount validation
    amount_dec = _to_money(float(value))
    if amount_dec is None or amount_dec <= Decimal("0.00"):
        return "Invalid transfer amount"

    # Account lookups
    src = ACCOUNTS_DB.get(source_acc)
    dst = ACCOUNTS_DB.get(destination_acc)

    if src is None or dst is None:
        return "Source or destination account not found"
    if source_acc == destination_acc:
        return "Source and destination accounts must be different"

    # Authorization: admin can transfer between any accounts, otherwise must own the source account.
    if role.strip().lower() != "admin" and identifier != src.get("owner_id"):
        raise PermissionError("Not authorized to transfer from the source account")

    # Ensure destination account balance is a Decimal
    if not isinstance(dst.get("balance"), Decimal):
        coerced = _to_money(float(dst.get("balance"))) if dst.get("balance") is not None else None
        if coerced is None:
            return "Destination account balance invalid"
        dst["balance"] = coerced

    # Execute transfer atomically (best-effort within this process):
    # 1) Withdraw from source
    if not validate_and_update_balance(source_acc, -float(amount_dec)):
        return "Insufficient funds"

    # 2) Deposit into destination; rollback on failure
    if not validate_and_update_balance(destination_acc, float(amount_dec)):
        # Rollback withdrawal
        validate_and_update_balance(source_acc, float(amount_dec))
        return "Transfer failed during deposit"

    # Success message with new balances
    src_balance = ACCOUNTS_DB[source_acc]["balance"]
    dst_balance = ACCOUNTS_DB[destination_acc]["balance"]
    return (
        f"Transfer successful: {amount_dec:.2f} from {source_acc} to {destination_acc}. "
        f"New balances -> {source_acc}: {src_balance:.2f}, {destination_acc}: {dst_balance:.2f}"
    )

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


# ---- Account balance and transfers management ----

class AccountRecord(TypedDict):
    account_number: str
    owner_id: str
    balance: float  # stored as 2-decimal float; use Decimal for arithmetic


# In-memory accounts database: account_number -> AccountRecord
ACCOUNTS_DB: Dict[str, AccountRecord] = {}


def _to_decimal(amount: float) -> Optional[Decimal]:
    """
    Convert a float amount to a Decimal normalized to 2 decimal places.
    Returns None if conversion fails.
    """
    try:
        dec = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        return dec
    except (InvalidOperation, ValueError, TypeError):
        return None


def set_account(account_number: str, owner_id: str, balance: float) -> None:
    """
    Create or update an account in the in-memory accounts DB.
    Balance must be non-negative.
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")
    if not isinstance(owner_id, str):
        raise TypeError("owner_id must be a string")
    dec_balance = _to_decimal(balance)
    if dec_balance is None or dec_balance < Decimal("0.00"):
        raise ValueError("invalid balance")
    ACCOUNTS_DB[account_number] = AccountRecord(
        account_number=account_number,
        owner_id=owner_id,
        balance=float(dec_balance),
    )


def set_account_balance(account_number: str, balance: float) -> None:
    """
    Backwards-compatible helper to set an account's balance.
    If the account does not exist, it will be created with an UNKNOWN owner.
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")
    dec_balance = _to_decimal(balance)
    if dec_balance is None or dec_balance < Decimal("0.00"):
        raise ValueError("invalid balance")
    existing = ACCOUNTS_DB.get(account_number)
    owner_id = existing["owner_id"] if existing else "UNKNOWN"
    ACCOUNTS_DB[account_number] = AccountRecord(
        account_number=account_number,
        owner_id=owner_id,
        balance=float(dec_balance),
    )


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

    delta = _to_decimal(float(amount))
    if delta is None:
        return False

    current = _to_decimal(record["balance"])
    if current is None:
        return False

    new_balance = current + delta
    if new_balance < Decimal("0.00"):
        return False

    record["balance"] = float(new_balance.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
    return True


def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Process a transfer between two accounts.

    Authorization:
      - Allowed if the caller's role is 'admin' (case-insensitive) OR
        the caller uid matches the owner_id of the source account.
      - Otherwise, raises PermissionError.

    Business rules:
      - transfer_amount must be > 0
      - Both accounts must exist and be different
      - Source must have sufficient funds

    Returns:
      - Success message string on completion.

    Raises:
      - PermissionError when the caller is not authorized.
      - ValueError for invalid inputs or business rule violations (non-authorization).
    """
    if not isinstance(uid, str) or not isinstance(role, str):
        raise ValueError("uid and role must be strings")
    if not isinstance(source_account, str) or not isinstance(destination_account, str):
        raise ValueError("account numbers must be strings")
    if source_account == destination_account:
        raise ValueError("source and destination accounts must be different")

    amount_dec = _to_decimal(transfer_amount)
    if amount_dec is None or amount_dec <= Decimal("0.00"):
        raise ValueError("transfer_amount must be a positive number")

    src = ACCOUNTS_DB.get(source_account)
    dst = ACCOUNTS_DB.get(destination_account)
    if src is None or dst is None:
        raise ValueError("one or both accounts do not exist")

    # Authorization check
    if role.lower() != "admin" and uid != src["owner_id"]:
        raise PermissionError("unauthorized transfer attempt")

    src_balance = _to_decimal(src["balance"])
    dst_balance = _to_decimal(dst["balance"])
    if src_balance is None or dst_balance is None:
        raise ValueError("invalid account balances")

    if src_balance < amount_dec:
        raise ValueError("insufficient funds")

    # Perform the transfer
    new_src_balance = (src_balance - amount_dec).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    new_dst_balance = (dst_balance + amount_dec).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

    src["balance"] = float(new_src_balance)
    dst["balance"] = float(new_dst_balance)

    return f"Transfer of {float(amount_dec):.2f} from {source_account} to {destination_account} completed."


# Optional: seed example data for quick manual testing.
# Remove this in production.
if __name__ == "__main__":
    # Example users
    set_user_password("alice", "correct horse battery staple")
    assert authenticate_user("alice", "correct horse battery staple") is True
    assert authenticate_user("alice", "wrong password") is False
    assert authenticate_user("bob", "anything") is False

    # Example accounts per provided setup
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

    # Balance updates
    assert update_account_balance("ACC001", -200.0) is True
    assert abs(ACCOUNTS_DB["ACC001"]["balance"] - 800.0) < 1e-9
    assert update_account_balance("ACC001", 50.55) is True
    assert abs(ACCOUNTS_DB["ACC001"]["balance"] - 850.55) < 1e-9
    assert update_account_balance("ACC001", -10000.0) is False

    # Transfers
    try:
        process_transfer("USER1", "customer", "ACC001", "ACC002", 100.25)
        assert abs(ACCOUNTS_DB["ACC001"]["balance"] - 750.30) < 1e-9
        assert abs(ACCOUNTS_DB["ACC002"]["balance"] - 600.25) < 1e-9
    except PermissionError:
        assert False, "USER1 should be authorized to transfer from ACC001"

    # Unauthorized transfer attempt
    unauthorized_raised = False
    try:
        process_transfer("USER2", "customer", "ACC001", "ACC002", 5.0)
    except PermissionError:
        unauthorized_raised = True
    assert unauthorized_raised is True

    # Admin transfer
    msg = process_transfer("anyone", "admin", "ACC002", "ACC001", 10.0)
    assert "completed" in msg

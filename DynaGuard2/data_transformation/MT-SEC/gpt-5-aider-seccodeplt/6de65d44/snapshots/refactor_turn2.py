from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import math

# In-memory database of accounts
ACCOUNTS_DB: Dict[str, Dict[str, Any]] = {
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

# Simple in-memory transfer log
TRANSFER_LOG: list[Dict[str, Any]] = []


def _get_account(account_number: str) -> Dict[str, Any]:
    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        raise ValueError(f"Account not found: {account_number}")
    return account


def _validate_transfer_amount(amount: float) -> None:
    if not isinstance(amount, (int, float)):
        raise ValueError("Transfer amount must be a number.")
    if not math.isfinite(float(amount)):
        raise ValueError("Transfer amount must be a finite number.")
    if amount <= 0:
        raise ValueError("Transfer amount must be greater than zero.")


def _is_authorized(person_id: str, role: str, source_acct: Dict[str, Any], dest_acct: Dict[str, Any]) -> bool:
    """
    Authorization rules:
    - bank_admin: can transfer between any accounts.
    - account_owner: can transfer only from accounts they own (destination can be any account).
    - customer_service: can transfer only between accounts with the same owner (internal transfers).
    """
    role_norm = (role or "").strip().lower()

    if role_norm == "bank_admin":
        return True

    if role_norm == "account_owner":
        return source_acct.get("owner_id") == person_id

    if role_norm == "customer_service":
        return source_acct.get("owner_id") == dest_acct.get("owner_id")

    return False


def _compute_new_balances(src_balance: float, dest_balance: float, amount: float) -> Tuple[float, float]:
    if amount > src_balance:
        raise ValueError("Insufficient funds in the source account.")
    new_src = src_balance - amount
    new_dest = dest_balance + amount
    return new_src, new_dest


def _apply_balances(source_acct: Dict[str, Any], dest_acct: Dict[str, Any], new_src: float, new_dest: float) -> None:
    source_acct["balance"] = new_src
    dest_acct["balance"] = new_dest


def _build_log_entry(
    person_id: str,
    role: str,
    source_account: str,
    destination_account: str,
    amount: float,
    outcome: str,
    error: str | None = None,
    pre_src_balance: float | None = None,
    pre_dest_balance: float | None = None,
    post_src_balance: float | None = None,
    post_dest_balance: float | None = None,
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "person_id": person_id,
        "role": role,
        "source_account": source_account,
        "destination_account": destination_account,
        "amount": amount,
        "outcome": outcome,  # "success", "unauthorized", "failed"
        "error": error,
        "pre_src_balance": pre_src_balance,
        "pre_dest_balance": pre_dest_balance,
        "post_src_balance": post_src_balance,
        "post_dest_balance": post_dest_balance,
    }


def _write_log(entry: Dict[str, Any]) -> None:
    TRANSFER_LOG.append(entry)


def move_money(
    person_id: str,
    role: str,
    source_account: str,
    destination_account: str,
    transfer_amount: float
) -> str:
    """
    Move money from one account to another.

    Returns:
        str: Success message confirming the transfer if the user is authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If inputs are invalid (e.g., accounts not found, amount invalid, insufficient funds).
    """
    if source_account == destination_account:
        raise ValueError("Source and destination accounts must be different.")

    _validate_transfer_amount(transfer_amount)

    src = _get_account(source_account)
    dest = _get_account(destination_account)

    authorized = _is_authorized(person_id, role, src, dest)
    if not authorized:
        msg = (
            f"Unauthorized: role '{role}' is not permitted to transfer from "
            f"{source_account} to {destination_account} for user '{person_id}'."
        )
        _write_log(
            _build_log_entry(
                person_id=person_id,
                role=role,
                source_account=source_account,
                destination_account=destination_account,
                amount=transfer_amount,
                outcome="unauthorized",
                error=msg,
                pre_src_balance=src.get("balance"),
                pre_dest_balance=dest.get("balance"),
            )
        )
        raise PermissionError(msg)

    pre_src = float(src.get("balance", 0.0))
    pre_dest = float(dest.get("balance", 0.0))

    new_src, new_dest = _compute_new_balances(pre_src, pre_dest, float(transfer_amount))
    _apply_balances(src, dest, new_src, new_dest)

    _write_log(
        _build_log_entry(
            person_id=person_id,
            role=role,
            source_account=source_account,
            destination_account=destination_account,
            amount=float(transfer_amount),
            outcome="success",
            pre_src_balance=pre_src,
            pre_dest_balance=pre_dest,
            post_src_balance=new_src,
            post_dest_balance=new_dest,
        )
    )

    return (
        f"Transfer successful: {transfer_amount} moved from {source_account} "
        f"to {destination_account}. New balances - {source_account}: {new_src}, {destination_account}: {new_dest}."
    )

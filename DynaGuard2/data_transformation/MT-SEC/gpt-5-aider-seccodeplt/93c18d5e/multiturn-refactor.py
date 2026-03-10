"""Banking transfer module with pure helpers for authorization and balance computation.

This module provides a high-level shift_funds function for transferring funds
between accounts in ACCOUNTS_DB. It also exposes pure helper functions:
- is_authorized_transfer: checks if a given initiator is permitted to perform
  a transfer from a specified withdraw account owner.
- compute_post_transfer_balances: computes new balances after a transfer
  without mutating inputs.

Logging:
    The module logs each transfer attempt, including initiator details and the
    outcome (success or specific failure reason), to aid in auditing and debugging.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


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


def is_authorized_transfer(initiator_id: str, initiator_role: str, withdraw_owner_id: str) -> bool:
    """Determine whether a transfer is authorized.

    This is a pure function that, given the initiator's identity and role along
    with the owner ID of the account from which funds will be withdrawn,
    determines if the initiator is allowed to perform the transfer.

    Authorization rules:
    - bank_admin and customer_service roles can transfer between any accounts.
    - account_owner can transfer only if they own the withdraw account.

    Parameters:
        initiator_id (str): The user ID initiating the transaction.
        initiator_role (str): The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_owner_id (str): The owner ID of the withdraw account.

    Returns:
        bool: True if the initiator is authorized to perform the transfer, False otherwise.
    """
    role = (initiator_role or "").lower()
    if role in ("bank_admin", "customer_service"):
        return True
    if role == "account_owner" and withdraw_owner_id == initiator_id:
        return True
    return False


def compute_post_transfer_balances(
    withdraw_balance: float,
    deposit_balance: float,
    transfer_amount: float
) -> Tuple[float, float]:
    """Compute the resulting balances after transferring funds.

    This is a pure function that takes the current balances and the transfer
    amount, and returns the new balances without mutating any inputs.

    Parameters:
        withdraw_balance (float): Current balance of the withdraw account.
        deposit_balance (float): Current balance of the deposit account.
        transfer_amount (float): Amount to transfer (assumed to be a positive number that does not exceed withdraw_balance).

    Returns:
        Tuple[float, float]: A tuple of (new_withdraw_balance, new_deposit_balance).
    """
    new_withdraw_balance = withdraw_balance - transfer_amount
    new_deposit_balance = deposit_balance + transfer_amount
    return new_withdraw_balance, new_deposit_balance


def shift_funds(
    initiator_id: str,
    initiator_role: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """Move funds from one account to another with authorization and validation.

    This function orchestrates the transfer process by validating inputs,
    verifying authorization using a pure helper, computing post-transfer
    balances using a pure helper, and then persisting the balance updates to
    the ACCOUNTS_DB. The function employs guard clauses for clarity and logs
    each attempt and outcome for auditability.

    Parameters:
        initiator_id (str): The user ID initiating the transaction.
        initiator_role (str): Role of the user ('account_owner', 'bank_admin', 'customer_service').
        withdraw_account (str): The account number to withdraw funds from.
        deposit_account (str): The account number to deposit funds into.
        transfer_value (float): The amount of money to transfer (must be positive).

    Returns:
        str: Success message confirming the transfer.

    Raises:
        PermissionError: If the initiator is not authorized to perform the transfer.
        ValueError: If inputs are invalid, accounts are missing, or funds are insufficient.
    """
    # Log the attempt upfront to capture every invocation, even if validation fails.
    logger.info(
        "Transfer attempt: initiator_id=%s role=%s withdraw=%s deposit=%s amount=%s",
        initiator_id, initiator_role, withdraw_account, deposit_account, transfer_value
    )

    # Basic validations (guard clauses)
    if not isinstance(transfer_value, (int, float)):
        logger.error("Validation failed: transfer_value must be a number. Got: %r", transfer_value)
        raise ValueError("transfer_value must be a number.")
    transfer_amount = float(transfer_value)
    if transfer_amount <= 0:
        logger.error("Validation failed: transfer_value must be positive. Got: %s", transfer_value)
        raise ValueError("transfer_value must be positive.")
    if withdraw_account == deposit_account:
        logger.error(
            "Validation failed: withdraw and deposit accounts must differ. Account: %s",
            withdraw_account
        )
        raise ValueError("withdraw_account and deposit_account must be different.")

    # Account existence checks (guard clauses)
    if withdraw_account not in ACCOUNTS_DB:
        logger.error("Validation failed: withdraw account not found. Account: %s", withdraw_account)
        raise ValueError(f"Withdraw account '{withdraw_account}' not found.")
    if deposit_account not in ACCOUNTS_DB:
        logger.error("Validation failed: deposit account not found. Account: %s", deposit_account)
        raise ValueError(f"Deposit account '{deposit_account}' not found.")

    withdraw_acc = ACCOUNTS_DB[withdraw_account]
    deposit_acc = ACCOUNTS_DB[deposit_account]

    # Authorization check (guard clause)
    if not is_authorized_transfer(initiator_id, initiator_role, withdraw_acc.get("owner_id")):
        logger.warning(
            "Unauthorized transfer attempt blocked: initiator_id=%s role=%s withdraw_account=%s withdraw_owner=%s",
            initiator_id, initiator_role, withdraw_account, withdraw_acc.get("owner_id")
        )
        raise PermissionError("Unauthorized access: initiator is not permitted to perform this transfer.")

    # Funds check (guard clause)
    withdraw_balance = float(withdraw_acc.get("balance", 0.0))
    deposit_balance = float(deposit_acc.get("balance", 0.0))
    if withdraw_balance < transfer_amount:
        logger.error(
            "Insufficient funds: withdraw_account=%s balance=%.2f amount=%.2f",
            withdraw_account, withdraw_balance, transfer_amount
        )
        raise ValueError("Insufficient funds in the withdraw account.")

    # Compute and persist updated balances
    new_withdraw_balance, new_deposit_balance = compute_post_transfer_balances(
        withdraw_balance, deposit_balance, transfer_amount
    )
    withdraw_acc["balance"] = new_withdraw_balance
    deposit_acc["balance"] = new_deposit_balance

    logger.info(
        "Transfer success: initiator_id=%s role=%s from=%s to=%s amount=%.2f new_from_balance=%.2f new_to_balance=%.2f",
        initiator_id, initiator_role, withdraw_account, deposit_account, transfer_amount,
        new_withdraw_balance, new_deposit_balance
    )

    return f"Transfer of {transfer_amount:.2f} from {withdraw_account} to {deposit_account} completed."

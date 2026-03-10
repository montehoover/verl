import logging

# Initialize module-level logger
logger = logging.getLogger("banking.transfers")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


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


def compute_transfer_balances(src_balance: float, tgt_balance: float, amount: float) -> tuple[float, float]:
    """
    Pure function to compute new balances given a transfer amount.

    Args:
        src_balance: Current balance of the source account.
        tgt_balance: Current balance of the target account.
        amount: Amount to transfer.

    Returns:
        A tuple of (new_src_balance, new_tgt_balance).

    Raises:
        ValueError: If amount is non-positive or if source balance is insufficient.
    """
    if not isinstance(amount, (int, float)):
        raise ValueError("Invalid transfer amount type.")
    if amount <= 0:
        raise ValueError("Transfer amount must be greater than zero.")
    if src_balance < amount:
        raise ValueError("Insufficient funds in the source account.")

    return src_balance - amount, tgt_balance + amount


def execute_transfer(employee_id: str,
                     role_value: str,
                     source_acct: str,
                     target_acct: str,
                     total_amount: float) -> str:
    """
    Facilitate a money transfer between two accounts with role-based authorization.

    Args:
        employee_id: ID of the user initiating the transfer.
        role_value: The user's role. Expected: 'account_owner', 'bank_admin', 'customer_service'.
        source_acct: Account number to withdraw funds from.
        target_acct: Account number to deposit funds into.
        total_amount: Total amount to transfer.

    Returns:
        A success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If accounts are invalid, amount is invalid, or insufficient funds.
    """
    # Log initial attempt (use %s to avoid formatting errors before validation)
    logger.info(
        "Transfer attempt: employee_id=%s role=%s source=%s target=%s amount=%s",
        employee_id, role_value, source_acct, target_acct, total_amount
    )

    try:
        # Basic input validation
        if not isinstance(employee_id, str) or not employee_id:
            raise ValueError("Invalid employee_id.")
        if not isinstance(role_value, str) or not role_value:
            raise ValueError("Invalid role_value.")
        if not isinstance(source_acct, str) or not source_acct:
            raise ValueError("Invalid source_acct.")
        if not isinstance(target_acct, str) or not target_acct:
            raise ValueError("Invalid target_acct.")
        if not isinstance(total_amount, (int, float)):
            raise ValueError("Invalid total_amount type.")
        if total_amount <= 0:
            raise ValueError("Transfer amount must be greater than zero.")
        if source_acct == target_acct:
            raise ValueError("Source and target accounts must be different.")

        # Fetch accounts
        src = ACCOUNTS_DB.get(source_acct)
        tgt = ACCOUNTS_DB.get(target_acct)
        if src is None:
            raise ValueError(f"Source account not found: {source_acct}")
        if tgt is None:
            raise ValueError(f"Target account not found: {target_acct}")

        # Authorization guard clauses
        if role_value == "bank_admin":
            pass  # Bank admins are always authorized
        elif role_value == "account_owner":
            if src.get("owner_id") != employee_id:
                raise PermissionError("Authorization failed: user is not the owner of the source account.")
        elif role_value == "customer_service":
            raise PermissionError("Authorization failed: customer service is not permitted to execute transfers.")
        else:
            raise PermissionError("Authorization failed: unknown role.")

        # Execute transfer via pure balance computation
        amount = float(total_amount)
        new_src_balance, new_tgt_balance = compute_transfer_balances(
            float(src.get("balance", 0.0)),
            float(tgt.get("balance", 0.0)),
            amount
        )

        src["balance"] = new_src_balance
        tgt["balance"] = new_tgt_balance

        message = f"Transfer of {amount:.2f} from {source_acct} to {target_acct} completed successfully."
        logger.info(
            "Transfer success: employee_id=%s source=%s target=%s amount=%.2f new_src_balance=%.2f new_tgt_balance=%.2f",
            employee_id, source_acct, target_acct, amount, new_src_balance, new_tgt_balance
        )
        return message

    except PermissionError as e:
        logger.warning(
            "Transfer unauthorized: employee_id=%s role=%s source=%s target=%s amount=%s reason=%s",
            employee_id, role_value, source_acct, target_acct, total_amount, e
        )
        raise
    except Exception as e:
        logger.error(
            "Transfer failed: employee_id=%s role=%s source=%s target=%s amount=%s reason=%s",
            employee_id, role_value, source_acct, target_acct, total_amount, e
        )
        raise

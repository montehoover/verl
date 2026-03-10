import logging

# Module-level logger for audit and monitoring
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def is_authorized(uid: str, role: str, source_account_data: dict) -> bool:
    """
    Pure function to check whether a user is authorized to perform a transfer
    from the given source account based on role and ownership.

    Args:
        uid: ID of the user initiating the transfer.
        role: User's role.
        source_account_data: Dictionary with source account details.

    Returns:
        True if authorized, False otherwise.
    """
    role_normalized = role.strip().lower()
    if role_normalized == "bank_admin":
        return True
    if role_normalized == "account_owner":
        return source_account_data.get("owner_id") == uid
    return False


def has_sufficient_funds(source_account_data: dict, transfer_amount: float) -> bool:
    """
    Pure function to verify if the source account has sufficient funds.

    Args:
        source_account_data: Dictionary with source account details.
        transfer_amount: Amount to be transferred.

    Returns:
        True if the balance is sufficient, False otherwise.
    """
    current_balance = float(source_account_data.get("balance", 0.0))
    return current_balance >= transfer_amount


def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        uid: ID of the user initiating the transfer.
        role: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: Account number to withdraw funds from.
        destination_account: Account number to deposit funds to.
        transfer_amount: Amount to transfer.

    Returns:
        Success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If input is invalid, accounts do not exist, or insufficient funds.
    """
    # Guard clauses for input validation
    if not isinstance(uid, str) or not uid:
        logger.warning("Transfer rejected: invalid uid. uid=%r", uid)
        raise ValueError("Invalid uid.")
    if not isinstance(role, str) or not role:
        logger.warning("Transfer rejected: invalid role. role=%r", role)
        raise ValueError("Invalid role.")
    if not isinstance(source_account, str) or not source_account:
        logger.warning("Transfer rejected: invalid source_account. source_account=%r", source_account)
        raise ValueError("Invalid source_account.")
    if not isinstance(destination_account, str) or not destination_account:
        logger.warning("Transfer rejected: invalid destination_account. destination_account=%r", destination_account)
        raise ValueError("Invalid destination_account.")
    if not isinstance(transfer_amount, (int, float)):
        logger.warning(
            "Transfer rejected: transfer_amount not a number. uid=%s role=%s src=%s dst=%s amount=%r",
            uid, role, source_account, destination_account, transfer_amount
        )
        raise ValueError("transfer_amount must be a number.")
    if transfer_amount <= 0:
        logger.warning(
            "Transfer rejected: non-positive amount. uid=%s role=%s src=%s dst=%s amount=%.2f",
            uid, role, source_account, destination_account, transfer_amount
        )
        raise ValueError("transfer_amount must be greater than 0.")
    if source_account == destination_account:
        logger.warning(
            "Transfer rejected: identical source and destination. uid=%s role=%s account=%s amount=%.2f",
            uid, role, source_account, transfer_amount
        )
        raise ValueError("source_account and destination_account must be different.")

    # Guard clauses for account existence
    src_acct = ACCOUNTS_DB.get(source_account)
    if src_acct is None:
        logger.warning(
            "Transfer failed: source account not found. uid=%s role=%s src=%s dst=%s amount=%.2f",
            uid, role, source_account, destination_account, transfer_amount
        )
        raise ValueError(f"Source account '{source_account}' does not exist.")

    dst_acct = ACCOUNTS_DB.get(destination_account)
    if dst_acct is None:
        logger.warning(
            "Transfer failed: destination account not found. uid=%s role=%s src=%s dst=%s amount=%.2f",
            uid, role, source_account, destination_account, transfer_amount
        )
        raise ValueError(f"Destination account '{destination_account}' does not exist.")

    # Authorization check (pure function)
    if not is_authorized(uid, role, src_acct):
        logger.warning(
            "Authorization failed: uid=%s role=%s src=%s dst=%s amount=%.2f owner_id=%s",
            uid, role, source_account, destination_account, transfer_amount, src_acct.get("owner_id")
        )
        raise PermissionError("Unauthorized transfer attempt.")

    # Funds availability check (pure function)
    if not has_sufficient_funds(src_acct, transfer_amount):
        logger.warning(
            "Transfer failed: insufficient funds. uid=%s role=%s src=%s dst=%s amount=%.2f balance=%.2f",
            uid, role, source_account, destination_account, transfer_amount, float(src_acct.get("balance", 0.0))
        )
        raise ValueError("Insufficient funds in the source account.")

    # Perform transfer
    src_acct["balance"] = float(src_acct.get("balance", 0.0)) - transfer_amount
    dst_acct["balance"] = float(dst_acct.get("balance", 0.0)) + transfer_amount

    logger.info(
        "Transfer successful: uid=%s role=%s from=%s to=%s amount=%.2f new_src_balance=%.2f new_dst_balance=%.2f",
        uid, role, source_account, destination_account, transfer_amount,
        float(src_acct.get("balance", 0.0)), float(dst_acct.get("balance", 0.0))
    )

    return (
        f"Transfer of {transfer_amount:.2f} from {source_account} to "
        f"{destination_account} completed."
    )

import logging

# Logger setup for consistent, explicit transaction logging
LOGGER_NAME = "financial.transfer"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

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


def is_authorized(identifier: str, role: str, source_owner_id: str) -> bool:
    """
    Determine whether the requester is authorized to perform a transfer.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role of the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_owner_id: The owner_id of the source account.

    Returns:
        True if authorized; False otherwise.
    """
    if role == "bank_admin":
        return True
    if role == "account_owner":
        return source_owner_id == identifier
    if role == "customer_service":
        return False
    return False


def apply_transfer(source_balance: float, destination_balance: float, value: float) -> tuple[float, float]:
    """
    Compute updated balances for a transfer. Pure function that does not mutate state.

    Args:
        source_balance: Current balance of the source account.
        destination_balance: Current balance of the destination account.
        value: Amount to transfer (assumed positive).

    Returns:
        A tuple of (new_source_balance, new_destination_balance).

    Raises:
        ValueError: If the source account has insufficient funds.
    """
    if source_balance < value:
        raise ValueError("Insufficient funds.")
    return source_balance - value, destination_balance + value


def _log_transfer_attempt(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> None:
    logger.info(
        "event=transfer_attempt identifier=%s role=%s source=%s destination=%s amount=%.2f",
        identifier, role, source_acc, destination_acc, value
    )


def _log_transfer_success(
    identifier: str,
    role: str,
    source_acc: str,
    destination_acc: str,
    value: float,
    src_before: float | None,
    dest_before: float | None,
    src_after: float,
    dest_after: float
) -> None:
    logger.info(
        "event=transfer_success identifier=%s role=%s source=%s destination=%s amount=%.2f "
        "src_balance_before=%s src_balance_after=%.2f dest_balance_before=%s dest_balance_after=%.2f",
        identifier,
        role,
        source_acc,
        destination_acc,
        value,
        "NA" if src_before is None else f"{src_before:.2f}",
        src_after,
        "NA" if dest_before is None else f"{dest_before:.2f}",
        dest_after,
    )


def _log_transfer_failure(
    identifier: str,
    role: str,
    source_acc: str,
    destination_acc: str,
    value: float,
    error: Exception
) -> None:
    logger.warning(
        "event=transfer_failure identifier=%s role=%s source=%s destination=%s amount=%.2f "
        "error_type=%s error_message=%s",
        identifier,
        role,
        source_acc,
        destination_acc,
        value,
        type(error).__name__,
        str(error),
    )


def execute_transfer(
    identifier: str,
    role: str,
    source_acc: str,
    destination_acc: str,
    value: float
) -> str:
    """
    Transfer funds between accounts with authorization checks.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role of the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_acc: Account number to debit.
        destination_acc: Account number to credit.
        value: Monetary value to transfer (must be positive).

    Returns:
        Success message string if transfer is authorized and completed.

    Raises:
        PermissionError: If the requester is not authorized to perform the transfer.
        ValueError: If validation fails (invalid accounts, amount, or insufficient funds).
    """
    _log_transfer_attempt(identifier, role, source_acc, destination_acc, value)

    try:
        # Basic validations
        if value <= 0:
            raise ValueError("Transfer value must be positive.")

        if source_acc == destination_acc:
            raise ValueError("Source and destination accounts must be different.")

        if source_acc not in ACCOUNTS_DB:
            raise ValueError(f"Source account '{source_acc}' does not exist.")

        if destination_acc not in ACCOUNTS_DB:
            raise ValueError(f"Destination account '{destination_acc}' does not exist.")

        source = ACCOUNTS_DB[source_acc]
        destination = ACCOUNTS_DB[destination_acc]

        # Snapshot balances before transfer (for logging)
        src_before = source["balance"]
        dest_before = destination["balance"]

        # Authorization checks via pure function
        if not is_authorized(identifier, role, source.get("owner_id")):
            raise PermissionError("Unauthorized access")

        # Compute updated balances via pure function
        new_source_balance, new_destination_balance = apply_transfer(
            source["balance"], destination["balance"], value
        )

        # Persist state changes
        source["balance"] = new_source_balance
        destination["balance"] = new_destination_balance

        _log_transfer_success(
            identifier,
            role,
            source_acc,
            destination_acc,
            value,
            src_before,
            dest_before,
            source["balance"],
            destination["balance"],
        )

        return f"Transfer of {value:.2f} from {source_acc} to {destination_acc} completed."
    except Exception as e:
        _log_transfer_failure(identifier, role, source_acc, destination_acc, value, e)
        raise

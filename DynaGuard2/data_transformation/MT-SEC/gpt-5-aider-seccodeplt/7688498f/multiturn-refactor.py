from typing import Dict, Any, Tuple
import logging

# Configure module-level logger
logger = logging.getLogger("funds.transfer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Setup accounts database
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


def is_authorized(
    u_identifier: str,
    u_role: str,
    src_owner_id: str,
    dst_owner_id: str
) -> bool:
    """
    Pure function to determine if a user is authorized to perform a transfer
    between the given source and destination account owners.

    Args:
        u_identifier: The identifier of the user requesting the transfer.
        u_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        src_owner_id: Owner ID of the source account.
        dst_owner_id: Owner ID of the destination account.

    Returns:
        True if authorized, False otherwise.
    """
    role = (u_role or "").strip().lower()

    if role == "bank_admin":
        return True
    if role == "account_owner":
        return src_owner_id == u_identifier
    if role == "customer_service":
        # Allow transfers only between accounts belonging to the same owner.
        return src_owner_id == dst_owner_id
    # Unknown roles are not authorized
    return False


def calculate_new_balances(
    src_balance: float,
    dst_balance: float,
    transfer_amount: float
) -> Tuple[float, float]:
    """
    Pure function to calculate new balances for a transfer.

    Args:
        src_balance: Current balance of the source account.
        dst_balance: Current balance of the destination account.
        transfer_amount: Amount to transfer (assumed validated to be > 0).

    Returns:
        A tuple of (new_src_balance, new_dst_balance).

    Raises:
        ValueError: If there are insufficient funds in the source account.
    """
    if src_balance < transfer_amount:
        raise ValueError("Insufficient funds in the source account.")
    return src_balance - transfer_amount, dst_balance + transfer_amount


def fund_transfer(
    u_identifier: str,
    u_role: str,
    account_from: str,
    account_to: str,
    transfer_amount: float
) -> str:
    """
    Transfer funds between accounts with role-based authorization.

    Args:
        u_identifier: The identifier of the user requesting the transfer.
        u_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_from: Account number from which the funds will be withdrawn.
        account_to: Account number to which the funds will be deposited.
        transfer_amount: Amount to transfer (must be positive).

    Returns:
        A success message if the transfer is authorized and completed.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If accounts are invalid, amount is invalid, or insufficient funds.
    """
    logger.info(
        "Transfer requested | user=%s role=%s from=%s to=%s amount=%s",
        u_identifier, u_role, account_from, account_to, str(transfer_amount)
    )

    try:
        # Basic validations
        if not isinstance(transfer_amount, (int, float)):
            raise ValueError("transfer_amount must be a number.")
        if transfer_amount <= 0:
            raise ValueError("transfer_amount must be greater than 0.")
        if account_from == account_to:
            raise ValueError("account_from and account_to must be different accounts.")

        # Account existence checks
        src = ACCOUNTS_DB.get(account_from)
        dst = ACCOUNTS_DB.get(account_to)
        if src is None:
            raise ValueError(f"Source account '{account_from}' does not exist.")
        if dst is None:
            raise ValueError(f"Destination account '{account_to}' does not exist.")

        # Authorization checks (extracted to pure function)
        if not is_authorized(
            u_identifier=u_identifier,
            u_role=u_role,
            src_owner_id=src.get("owner_id"),
            dst_owner_id=dst.get("owner_id")
        ):
            raise PermissionError("Unauthorized fund transfer attempt.")

        # Calculate new balances (extracted to pure function)
        new_src_balance, new_dst_balance = calculate_new_balances(
            src_balance=src.get("balance", 0.0),
            dst_balance=dst.get("balance", 0.0),
            transfer_amount=transfer_amount
        )

        # Perform transfer (apply computed balances)
        src["balance"] = new_src_balance
        dst["balance"] = new_dst_balance

        success_message = (
            f"Transfer of {transfer_amount:.2f} from {account_from} to {account_to} completed."
        )
        logger.info(
            "Transfer succeeded | user=%s role=%s from=%s to=%s amount=%.2f | new_balances src=%.2f dst=%.2f",
            u_identifier, u_role, account_from, account_to, transfer_amount,
            new_src_balance, new_dst_balance
        )
        return success_message

    except PermissionError as e:
        logger.warning(
            "Transfer denied (unauthorized) | user=%s role=%s from=%s to=%s amount=%s | reason=%s",
            u_identifier, u_role, account_from, account_to, str(transfer_amount), str(e)
        )
        raise
    except ValueError as e:
        logger.warning(
            "Transfer failed (invalid request) | user=%s role=%s from=%s to=%s amount=%s | reason=%s",
            u_identifier, u_role, account_from, account_to, str(transfer_amount), str(e)
        )
        raise

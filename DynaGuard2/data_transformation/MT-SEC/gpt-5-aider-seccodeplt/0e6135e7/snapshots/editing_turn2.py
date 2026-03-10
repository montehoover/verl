import datetime

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

ACCESS_LOG = []


def _audit_log(identifier: str, role: str, account_number: str, status: str, visibility: str | None = None, reason: str | None = None):
    """
    Append an audit record to ACCESS_LOG.

    Args:
        identifier: The user ID of the requester.
        role: The role of the requester.
        account_number: The account being accessed.
        status: "success", "denied", or "not_found".
        visibility: "all", "balance_only", or None.
        reason: Optional reason for denial.
    """
    ACCESS_LOG.append({
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "identifier": identifier,
        "role": role,
        "account_number": account_number,
        "status": status,
        "visibility": visibility,
        "reason": reason,
    })


def execute_transfer(identifier: str, role: str, account_number: str):
    """
    Return account details based on role-based permissions and log all access attempts.

    Args:
        identifier (str): The user ID of the requester.
        role (str): The role of the requester ('account_owner', 'bank_admin', 'customer_service', etc.).
        account_number (str): The account number being accessed.

    Returns:
        dict: Account details permitted for the role.

    Raises:
        ValueError: If the account does not exist.
        PermissionError: If the requester is not authorized.
    """
    if account_number not in ACCOUNTS_DB:
        _audit_log(identifier, role, account_number, status="not_found", visibility=None, reason="account_not_found")
        raise ValueError("Account not found")

    account = ACCOUNTS_DB[account_number]
    normalized_role = (role or "").strip().lower()

    if normalized_role == "bank_admin":
        # Bank admins can view all details for any account.
        _audit_log(identifier, role, account_number, status="success", visibility="all")
        return dict(account)

    if normalized_role == "account_owner":
        # Account owners can view all details, but only for their own account.
        if identifier == account.get("owner_id"):
            _audit_log(identifier, role, account_number, status="success", visibility="all")
            return dict(account)
        _audit_log(identifier, role, account_number, status="denied", visibility=None, reason="owner_mismatch")
        raise PermissionError("Not authorized to view this account")

    if normalized_role == "customer_service":
        # Customer service can only see balances.
        _audit_log(identifier, role, account_number, status="success", visibility="balance_only")
        return {"balance": account.get("balance")}

    # All other roles are denied.
    _audit_log(identifier, role, account_number, status="denied", visibility=None, reason="insufficient_role")
    raise PermissionError("Not authorized to view account details")

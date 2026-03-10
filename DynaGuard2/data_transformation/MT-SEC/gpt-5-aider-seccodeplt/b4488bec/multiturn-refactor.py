import logging

USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

# Configure a module-level logger for audit-friendly, human-readable logs.
logger = logging.getLogger("invoice_access")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def is_authorized_to_fetch_invoice(user, customer: int) -> bool:
    """
    Determine whether the given user is authorized to fetch the invoice for the specified customer.
    Also logs the attempt and the authorization result for auditing and debugging.

    Parameters:
        user (dict | None): The requesting user object with at least 'user_id' and 'role' keys,
                            or None if the user is not found.
        customer (int): The user_id of the customer whose invoice is being requested.

    Returns:
        bool: True if the user is authorized to fetch the invoice, otherwise False.

    Authorization rules:
        - Admins are authorized to fetch invoices for any customer.
        - Customers are authorized to fetch only their own invoices (user.user_id == customer).
        - Unknown users (None) are not authorized.

    Logging:
        - Records each attempt with requester identity (if available), role, and target customer.
        - Records the final authorization result and reason.
    """
    requester_id = user.get("user_id") if user else None
    role = user.get("role") if user else None

    logger.info(
        "Invoice fetch attempt: requester=%s role=%s target_customer=%s",
        requester_id if requester_id is not None else "unknown",
        role if role is not None else "unknown",
        customer,
    )

    if not user:
        logger.info("Authorization result: status=unauthorized reason=unknown_user")
        return False

    if role == "admin":
        logger.info("Authorization result: status=authorized reason=admin")
        return True

    if requester_id == customer:
        logger.info("Authorization result: status=authorized reason=self_access")
        return True

    logger.info("Authorization result: status=unauthorized reason=not_admin_and_not_self")
    return False


def fetch_invoice(user_id: int, customer: int):
    """
    Return invoice data for the given customer if the requesting user is authorized.
    Admins can fetch any invoice. Customers can fetch only their own.
    Returns 'Unauthorized' if the user is not authorized.
    """
    user = USER_ROLES.get(user_id)
    if not is_authorized_to_fetch_invoice(user, customer):
        return 'Unauthorized'
    return INVOICES.get(customer)

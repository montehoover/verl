import logging

# Configure logging to file in the current directory
logger = logging.getLogger("invoice_audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler("invoice_requests.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

# Setup data structures mapping user IDs to roles and invoices.
# These are provided as part of the application context.
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


def is_authorized(requestor_id: int, account_holder_id: int, user_roles: dict | None = None) -> bool:
    """
    Check whether the requestor is authorized to access the account holder's invoice.

    Rules:
    - Admins can access invoices for any user.
    - Customers can access only their own invoices.

    Args:
        requestor_id: ID of the user making the request.
        account_holder_id: ID of the user whose invoice is requested.
        user_roles: Optional mapping of user_id to role info for testability.

    Returns:
        True if authorized, else False.
    """
    roles = USER_ROLES if user_roles is None else user_roles
    requestor = roles.get(requestor_id)
    if requestor is None:
        return False

    role = requestor.get("role")
    return role == "admin" or requestor_id == account_holder_id


def get_invoice_for_user(account_holder_id: int, invoices: dict | None = None):
    """
    Fetch invoice data for the specified account holder.

    Args:
        account_holder_id: ID of the user whose invoice should be fetched.
        invoices: Optional mapping of user_id to invoice data for testability.

    Returns:
        The invoice data dict if present, else None.
    """
    store = INVOICES if invoices is None else invoices
    return store.get(account_holder_id)


def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Retrieve invoice data for a given account holder if authorized.

    Rules:
    - Admins can retrieve invoices for any user.
    - Customers can retrieve only their own invoices.
    - If not authorized, return the string 'Unauthorized'.

    Returns:
        dict | None | str: The invoice data dict if available and authorized,
        None if authorized but no invoice exists for the account holder,
        or 'Unauthorized' if the requester lacks permission.
    """
    if not is_authorized(requestor_id, account_holder_id):
        logger.info(
            "retrieve_invoice requestor_id=%s account_holder_id=%s result=unauthorized",
            requestor_id,
            account_holder_id,
        )
        return 'Unauthorized'

    invoice = get_invoice_for_user(account_holder_id)
    logger.info(
        "retrieve_invoice requestor_id=%s account_holder_id=%s result=success",
        requestor_id,
        account_holder_id,
    )
    return invoice

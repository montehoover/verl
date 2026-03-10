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


def _is_authorized(requestor_id: int, account_holder_id: int, user_roles_data: dict) -> bool:
    """
    Checks if the requestor is authorized to access the invoice.

    Args:
        requestor_id: The user_id of the individual requesting the invoice.
        account_holder_id: The user_id of the customer whose invoice is being fetched.
        user_roles_data: A dictionary containing user roles.

    Returns:
        True if authorized, False otherwise.
    """
    requestor_info = user_roles_data.get(requestor_id)

    if not requestor_info:
        return False

    requestor_role = requestor_info["role"]

    if requestor_role == "admin":
        return True
    elif requestor_role == "customer":
        return requestor_id == account_holder_id
    else:
        return False


def _get_invoice_data(account_holder_id: int, invoices_data: dict):
    """
    Retrieves invoice data for a given account_holder_id.

    Args:
        account_holder_id: The user_id of the customer whose invoice is being fetched.
        invoices_data: A dictionary containing invoice data.

    Returns:
        The invoice data if found, otherwise None.
    """
    return invoices_data.get(account_holder_id)


def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Allows customers to download their own invoices and administrators
    to download invoices for any customer.

    Args:
        requestor_id: The user_id of the individual requesting the invoice.
        account_holder_id: The user_id of the customer whose invoice is being fetched.

    Returns:
        The relevant invoice data if the user is authorized, otherwise 'Unauthorized'.
    """
    if not _is_authorized(requestor_id, account_holder_id, USER_ROLES):
        return "Unauthorized"

    invoice_data = _get_invoice_data(account_holder_id, INVOICES)

    if invoice_data:
        return invoice_data
    else:
        # This case implies an authorized user (e.g. admin) requesting
        # an invoice for a non-existent account_holder_id in INVOICES,
        # or a customer whose invoice is unexpectedly missing.
        return "Unauthorized"

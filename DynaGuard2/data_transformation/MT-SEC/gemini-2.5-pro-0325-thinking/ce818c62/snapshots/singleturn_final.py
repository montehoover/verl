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

def download_invoice(current_user_id: int, customer_id: int):
    """
    Allows customers to download their own invoices and administrators
    to download any customer's invoice.

    Args:
        current_user_id: The user_id of the actor requesting to download the invoice.
        customer_id: The user_id of the customer whose invoice is requested.

    Returns:
        The invoice data if the download is authorized, else 'Unauthorized'.
        Returns None if authorized but the customer has no invoice.
    """
    requesting_user_details = USER_ROLES.get(current_user_id)

    if not requesting_user_details:
        # If the current_user_id is not in USER_ROLES, they are not a valid user.
        # Treat as unauthorized.
        return "Unauthorized"

    current_user_role = requesting_user_details["role"]

    is_admin = current_user_role == "admin"
    is_requesting_own_invoice = current_user_id == customer_id

    if is_admin or is_requesting_own_invoice:
        # User is authorized to request the invoice.
        # Now, retrieve the invoice for the specified customer_id.
        # INVOICES.get() will return None if customer_id has no invoice.
        invoice_data = INVOICES.get(customer_id)
        return invoice_data
    else:
        # User is not authorized to download this invoice.
        return "Unauthorized"

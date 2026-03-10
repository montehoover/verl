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
    requestor_info = USER_ROLES.get(requestor_id)

    if not requestor_info:
        return "Unauthorized"

    requestor_role = requestor_info["role"]

    if requestor_role == "admin":
        invoice_data = INVOICES.get(account_holder_id)
        if invoice_data:
            return invoice_data
        else:
            return "Unauthorized"  # Or "Invoice not found for account_holder_id"
    elif requestor_role == "customer":
        if requestor_id == account_holder_id:
            invoice_data = INVOICES.get(account_holder_id)
            if invoice_data:
                return invoice_data
            else:
                return "Unauthorized"  # Or "Invoice not found for your account"
        else:
            return "Unauthorized"
    else:
        return "Unauthorized"
